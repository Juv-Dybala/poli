import argparse
import bitsandbytes as bnb
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, Pipeline
from transformers.data import default_data_collator
from datasets import load_dataset
import json
import copy
from eval import inference_eval, ckpt_eval, math_eval
from datasets_load import datasets_load,math_datasets_load
from llama_utils import get_math_prompt


QUESTION_PROMPT = {
    "no_opinion": 
        # "Question: {Question}. What do you think the answer is? Why? \n",
        "Question: {Question}. \n",
    "with_opinion": 
        "Question: {Question}. Opinion: I think the answer is ({Opinion}), what do you think about? Why? \n"
}
COT_PROMPT = "Please think step by step. \n"
ANSWER_PROMPT = "Answer: The correct answer is {Answer[0]} {Answer[1]}. \n {Rationale}"
MATH_ANSWER_PROMPT = "Answer: {Rationale} #### {Answer}"


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
        "--math",
        action="store_true",
        help="Whether fine-tune on math problem datdaset."
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        required=True,
        help="The alias directory name."
    )
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default="",
        help="The pretrain model directory."
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
        default='steps',
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
        "--eval_ckpts",
        action="store_true",
        help="Whether to eval ckpts after training."
    )
    parser.add_argument(
        "--eval_opinion",
        action="store_true",
        help="When evaluating on validation set, using with-opinion input."
    )
    return parser.parse_args()


class QADataset(Dataset):
    
    def __init__(self,dataset_name,dir_name,tokenizer,max_words=512):
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join("../data/finetuning",dataset_name,"{}.jsonl".format(dir_name))
        self.datas = self.load_data(self.dataset_dir)
        self.tokenizer = tokenizer
        self.max_words = max_words
        
    def load_data(self,dataset_dir):
        datafile = open(dataset_dir,mode="r")
        datas = []
        for line in datafile:
            datas.append(json.loads(line))
        return datas

    def __len__(self):
        return len(self.datas) 

    def __getitem__(self, index):
        item = self.datas[index]
        print(item)
        if "Opinion" in item:
            prompt = QUESTION_PROMPT["with_opinion"].format_map(item) + COT_PROMPT
        else:
            prompt = QUESTION_PROMPT["no_opinion"].format_map(item) + COT_PROMPT
        example = prompt + ANSWER_PROMPT.format_map(item)
        print(example)

        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat(
                (example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            'input_ids': example,
            'labels': labels,
            'attention_mask': example_mask,
        }


def create_prompt_formats(item,math=False):
    
    if "Opinion" in item:
        question = QUESTION_PROMPT["with_opinion"].format_map(item)
    else:
        question = QUESTION_PROMPT["no_opinion"].format_map(item)

    if math:
        answer = MATH_ANSWER_PROMPT.format_map(item)
        example = get_math_prompt(n_shot=4) + question + answer
    else:
        assert isinstance(item['Answer'],list) ,"Answer should have key and text."
        answer = ANSWER_PROMPT.format_map(item)
        # example = question + COT_PROMPT + answer
        example = question + answer
    
    item["text"] = example
    
    return item

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(dataset_name, dir_name, tokenizer, max_length, seed, math=False):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Load data
    print("Loading dataset:{}...".format(dataset_name))

    dataset_dir = os.path.join("../data/finetuning",dataset_name,"{}.jsonl".format(dir_name))
    datas = load_dataset('json',data_files=dataset_dir)
    
    # Add prompt to each sample
    print("Preprocessing dataset...")

    _prompt_function = partial(create_prompt_formats,math=math)
    dataset = datas.map(_prompt_function)
    
    remove_col = ["Question", "Answer", "Rationale", "text"]
    if "Opinion" in dataset.column_names:
        remove_col.append("Opinion")
    
    # Apply preprocessing to each batch of the dataset & and remove texts
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=remove_col,
    )

    # Filter out samples that have input_ids exceeding max_length
    if math:
        max_length = max_length + len(get_math_prompt(n_shot=4))
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
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

def train(model, tokenizer, dataset, log_dir, output_dir, args):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)
    # modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    print(modules)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules, args)
    model = get_peft_model(model, peft_config)
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)


    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
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
            output_dir=os.path.join("../log/SFT",args.dataset,log_dir),
            optim="paged_adamw_8bit",
            seed=args.seed,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     
    # Launch training
    print("Training...")
    
    
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)    
    
    ###
    
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


def eval(model,tokenizer,dataset_name,split='validation',opinion = False):

    eval_data = datasets_load(dataset_name,split=split)
    print(eval_data)
    
    result = inference_eval(model,tokenizer,eval_data,opinion)
    print("")
    print(result)
    return result
    

def eval_ckpts(dir_name,dataset_name,split='validation'):
    result = {}
    ckpt_dir = os.path.join("../log/SFT",dataset_name,dir_name)
    eval_data = datasets_load(dataset_name,split=split)
    
    for ckpt_name in next(os.walk(ckpt_dir))[1]:
        ckpt_path = os.path.join(ckpt_dir,ckpt_name)
        score = ckpt_eval(ckpt_path,eval_data)
        result[ckpt_name] = score
    return result


def eval_math(model, tokenizer, dataset_name, subset='math', split='test'):

    eval_data = math_datasets_load(dataset_name,subset=subset,split=split)
    print(eval_data)

    result = math_eval(model,tokenizer,eval_data,n_shot=8)
    print("")
    print(result)
    return result


def eval_math_ckpts(dir_name,dataset_name,subset='main',split='test'):
    result = {}
    ckpt_dir = os.path.join("../log/SFT",dataset_name,dir_name)
    eval_data = math_datasets_load(dataset_name,subset=subset,split=split)
    
    for ckpt_name in next(os.walk(ckpt_dir))[1]:
        ckpt_path = os.path.join(ckpt_dir,ckpt_name)
        score = ckpt_eval(ckpt_path,eval_data,math=True)
        result[ckpt_name] = score
    return result


if __name__ == "__main__":
    
    args = parse_args()
    print(args)
    
    model_name = args.model_name
    dataset_name = args.dataset
    dir_name = args.dir_name
    pretrain_dir = args.pretrain_dir

    output_dir = os.path.join("../result/lora_model/sft",dataset_name,dir_name)
    os.makedirs(output_dir,exist_ok=True)
    output_merged_dir = os.path.join("../result/sft_model",dataset_name,dir_name)
    
    original_model_save_directory = os.path.join("../models",model_name)
    pretrain_model_directory = os.path.join("../result/sft_model",dataset_name,pretrain_dir)
    if pretrain_dir != "" and os.path.exists(pretrain_model_directory):
        model_save_directory = pretrain_model_directory
    else:
        model_save_directory = original_model_save_directory
    print("dir:{}".format(model_save_directory))

    bnb_config = create_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(model_save_directory,
                                                 quantization_config=bnb_config,
                                                 device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_save_directory,use_auth_token=True)
    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    print(model)
    print(model.device)


    # 测评初始或已保存模型在数据集上的表现
    # if args.math:
    #     eval_math(model,tokenizer,dataset_name,subset='main',split='test')
    # else:
    #     eval(model,tokenizer,dataset_name,split="validation",opinion=args.eval_opinion)
    # exit()


    # dataset = QADataset(dataset_name,tokenizer).shuffle(seed=42)
    # dataloader = DataLoader(dataset,shuffle=True,batch_size=batch_size,
    #                         collate_fn=default_data_collator,drop_last=True)
    dataset = preprocess_dataset(dataset_name, dir_name, tokenizer,
                                 max_length=args.max_length,seed=args.seed,math=args.math)
    print(dataset)
    print(dataset['train'][0])
    
    train(model,tokenizer,dataset,dir_name,output_dir,args)

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
    if args.math:
        score = eval_math(model,tokenizer,dataset_name,subset='main',split='test')
    else:
        score = eval(model,tokenizer,dataset_name,split="validation",opinion=args.eval_opinion)

    if args.eval_ckpts:
        print("Evaluate ckpts...")
        if args.math:
            result = eval_math_ckpts(dir_name,dataset_name,subset='main',split='test')
        else:
            result = eval_ckpts(dir_name,dataset_name,split="validation")
        result['final'] = score
        print(result)
