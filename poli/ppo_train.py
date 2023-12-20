import torch
from transformers import AutoTokenizer,BitsAndBytesConfig,AutoModelForCausalLM
from trl import PPOConfig,PPOTrainer,AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import os
import argparse
from functools import partial
from tqdm import tqdm
from datasets_load import *
from fine_tune import eval
from pre_filter import extract_ar
from refined_selection import q2a,qr2a
from data_process import load_t5
import wandb


Q2A_PROMPT = "Question: {}. What do you think the answer is? Why? \nAnswer:"


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
        help="The alias directory name."
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="google/flan-t5-small",
        help="The name of reward model."
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Learning rate of fine-tune."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size of data."
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
        "--eval_opinion",
        action="store_true",
        help="When evaluating on validation set, using with-opinion input."
    )
    return parser.parse_args()


def build_dataset(dataset_name, tokenizer, max_length, seed):
    """
    Build dataset for training.
    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    # ds = load_ppo_data(dataset_name,dir_name)
    ds = datasets_load(dataset_name)
    # formatted_question、choices（text+label）、answerKey
   
    def tokenize(sample):
        sample['query'] = sample['formatted_question'] 
        answerKey = sample['answerKey']
        answerText = sample['choices']['text'][ord(answerKey)-ord('A')]
        sample['label'] = [f'({answerKey})',answerText]
        return sample

    ds = ds.map(tokenize, batched=False)

    ds = ds.filter(lambda sample: len(tokenizer.encode(sample["query"])) < max_length)
    ds = ds.shuffle(seed=seed)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


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


def ppo_train(args,ppo_config,model,model_ref,tokenizer,reward_model_name,dataset,output_dir):

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)
    print(modules)

    peft_config = create_peft_config(modules, args)
    model = get_peft_model(model, peft_config)
    # # 将lora模型加载入trl模型，加上value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model,device_map = "auto")
    print_trainable_parameters(model)

    ppo_trainer = PPOTrainer(ppo_config,model,model_ref,tokenizer,dataset,data_collator=collator)

    # ppo训练时关掉top k等限制采样空间的技术，也不设置最小长度
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens":args.max_length,
    }
    
    reward_model,reward_tokenizer = load_t5(reward_model_name)

    data = enumerate(ppo_trainer.dataloader)
    pbar = tqdm(total=len(ppo_trainer.dataloader))
    pbar.set_description("PPO Training...")

    for epoch, batch in data:
        print(f"epoch:{epoch}")
        
        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True

        questions = batch["query"]
        labels = batch["label"]
        batch_size = len(questions)
        
        query_tensors = []
        response_tensors = []
        response_list = []
        reward_list = []

        for index in range(batch_size):
            
            question = questions[index]
            true_answer = labels[index]
            
            query = torch.tensor(tokenizer.encode(Q2A_PROMPT.format(question))).to("cuda")
            query_tensors.append(query)

            response = ppo_trainer.generate(query,**generation_kwargs).squeeze()[query.size(0):]
            response_tensors.append(response)

            response_text = tokenizer.decode(response)
            response_list.append(response_text)

            answer,rationale = extract_ar(response_text)
            if answer != true_answer[0][1]:
                reward = -1.0 # 回答错误的reward值
            else:
                qr2a_prob = qr2a(reward_model,reward_tokenizer,question,true_answer,rationale,prob=True)
                q2a_prob = q2a(reward_model,reward_tokenizer,question,true_answer,prob=True)
                reward = qr2a_prob-q2a_prob
            reward = torch.tensor(reward)
            reward_list.append(reward)
            print(reward,end="  ")
            
        batch['response'] = response_list

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_list)
        ppo_trainer.log_stats(stats, batch, reward_list)

        pbar.update()
    
    pbar.close()
    # ppo_trainer.save_model()
    os.makedirs(output_dir,exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model,tokenizer


if __name__ == '__main__':

    args = parse_args()
    print(args)

    model_name = args.model_name
    dataset_name = args.dataset
    dir_name = args.dir_name
    reward_model_name = args.reward_model

    output_dir = os.path.join("../result/ppo_model",dir_name)
    
    original_model_save_directory = os.path.join("../models",model_name)
    pretrained_model_directory = os.path.join("../result/sft_model",dir_name)
    
    if os.path.exists(pretrained_model_directory):
        model_save_directory = pretrained_model_directory
    else:
        model_save_directory = original_model_save_directory

    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed = args.seed,
        log_with="wandb",
    )
    wandb.init(project="ppo")

    bnb_config = create_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_save_directory,
        quantization_config=bnb_config,
        device_map = "auto"
        )

    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_save_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(dataset_name,tokenizer,args.max_length,seed=args.seed)
    print(dataset)

    model,tokenizer = ppo_train(args,ppo_config,model,model_ref,tokenizer,reward_model_name,dataset,output_dir)

    print("Evaluate model...")
    eval(model,tokenizer,dataset_name,split="validation",opinion=args.eval_opinion)