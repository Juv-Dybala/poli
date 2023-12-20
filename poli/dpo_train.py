import torch
from transformers import AutoTokenizer,BitsAndBytesConfig,AutoModelForCausalLM,TrainingArguments
from trl import DPOTrainer
from trl.core import LengthSampler
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import os
import argparse
from tqdm import tqdm
from datasets_load import *
from fine_tune import eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The name of large model, which will be fine-tuned.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of dataset. Using train subset to fine-tune and validation subset to eval."
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        required=True,
        help="The alias directory name."
    )
    parser.add_argument(
        "--sft_dir",
        type=str,
        help="The supervise fine-tune model directory."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
        default=1024,
        help="Maximum number of tokens to emit from tokenizer."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        default=42,
        help="Random seed when shuffle the data and init training."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Temperature of DPO, the less beta is, the more we ignore ref-model."
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Rank(Dimesion) of LoRA."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="Scaling parameter of LoRA."
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for layers in LoRA."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size of data."
    )
    parser.add_argument(
        "--grad_acc_step",
        type=int,
        required=True,
        help="Step of accumulate gradient."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=-1,
        help="Warm up steps."
    )
    parser.add_argument(
        "--train_epoch",
        type=int,
        required=True,
        help="The number of train epochs."
    )
    parser.add_argument(
        "--max_step",
        type=int,
        help="The max step of fine-tune."
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        help="Save ckpts after each epoch or every n steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save ckpts after these steps."
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Learning rate of fine-tune."
    )
    parser.add_argument(
        "--eval_opinion",
        action="store_true",
        help="When evaluating on validation set, using with-opinion input."
    )
    return parser.parse_args()

    
def build_dataset(dataset_name, dir_name, seed):
    """Load the dataset
    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str], <Question>
        'chosen': List[str], <Good answer/rationale>
        'rejected': List[str], <Bad answer/rationale>
    }
    """
    dataset = load_dpo_data(dataset_name,dir_name)

    # def _data_process(sample):
    #     return {
    #         "prompt": sample['prompt'],
    #         "chosen": sample['chosen'],
    #         "rejected": sample['rejected'],
    #     }
    # dataset = dataset.map(_data_process)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def create_peft_config(modules, args):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=args.lora_r,  # dimension of the updated matrices
        lora_alpha=args.lora_alpha,  # parameter for scaling
        target_modules=modules,
        lora_dropout=args.lora_dropout,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            n = names[0] if len(names) == 1 else names[-1]
            lora_module_names.add(n)

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def dpo_train(model, model_ref, tokenizer, dataset, log_dir, output_dir, args):

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)
    print(modules)

    peft_config = create_peft_config(modules, args)
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # Training parameters
    # default data_collator: DPODataCollatorWithPadding
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_length=args.max_length,
        beta=args.beta,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size, # actual batch_size=micro_batch_size*grad_acc_step
            gradient_accumulation_steps=args.grad_acc_step, 
            warmup_steps=args.warmup,
            # max_steps=args.max_step,
            num_train_epochs=args.train_epoch,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=1,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            output_dir=os.path.join("../log/DPO",log_dir),
            optim="paged_adamw_8bit",
            seed=args.seed,
        ),
    )
    
    dpo_trainer.train()
    
    # dpo_trainer.save_model(output_dir)
    os.makedirs(output_dir,exist_ok=True)
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Free memory for merging weights
    del dpo_trainer
    torch.cuda.empty_cache()

    return model,tokenizer


if __name__ == '__main__':

    args = parse_args()
    print(args)

    model_name = args.model_name
    dataset_name = args.dataset
    dir_name = args.dir_name
    sft_dir = args.sft_dir

    output_dir = os.path.join("../result/lora_model/dpo",dir_name)
    output_merged_dir = os.path.join("../result/dpo_model",dir_name)
    
    original_model_save_directory = os.path.join("../models",model_name)
    sft_model_directory = os.path.join("../result/sft_model",dir_name)
    if os.path.exists(sft_model_directory):
        model_save_directory = sft_model_directory
    else:
        model_save_directory = original_model_save_directory

    print(model_save_directory)
    bnb_config = create_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_save_directory,
        quantization_config=bnb_config,
        device_map = "auto"
        )
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_save_directory,
        quantization_config=bnb_config,
        device_map = "auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(dataset_name,dir_name,seed=args.seed)
    print(dataset)

    model,tokenizer = dpo_train(model, model_ref, tokenizer, dataset,
                                log_dir=dir_name, output_dir=output_dir, args=args)

    os.makedirs(output_merged_dir,exist_ok=True)
    print("Merging LoRA and Saving model...")
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    # 此处输入的是PEFT model的dir，base model的地址在dir内的config中记录了
    print(model)
    model = model.merge_and_unload() # 将PEFT模型的参数合并到基础模型中，并释放PEFT模型的内存空间
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    print("=======================================")
    print(model)
    tokenizer.save_pretrained(output_merged_dir)
    
    print("Evaluate model...")
    score = eval(model,tokenizer,dataset_name,split="validation",opinion=args.eval_opinion)
    print(score)