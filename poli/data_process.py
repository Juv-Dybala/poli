import torch
import transformers
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM,T5ForConditionalGeneration
from datasets import load_dataset
import os.path
import re
from pre_filter import generate_answer
from refined_selection import select_rationale
import json
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--large_model",
        type=str,
        required=True,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The name of large model, which will be fine-tuned.",
    )
    parser.add_argument(
        "--small_model",
        type=str,
        required=True,
        default="google/flan-t5-small",
        help="The name of small model, which is used as discriminator.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of dataset. Defult Train subset."
    )
    parser.add_argument(
        "--inference_num",
        type=int,
        required=True,
        help="The total number of inference, including no-opinion input and with-opinion input."
    )
    parser.add_argument(
        "--wo_opinion_rate",
        type=float,
        required=True,
        help="The rate of no-opinion input in total inference number."
    )
    parser.add_argument(
        "--step2",
        action="store_true",
        help="Whether to process step2.(Using small LM to filter rationales.)"
    )
    parser.add_argument(
        "--use_opinion_ft",
        action="store_true",
        help="When generating fine-tuning data, using opinion input."
    )
    parser.add_argument(
        "--sycophancy",
        action="store_true",
        help="When generating fine-tuning data, using sycophancy distribution."
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        help="When generating processed data, the alias directory name."
    )
    return parser.parse_args()


def load_llama(model_name):
    
    model_save_directory = os.path.join("../models",model_name) 

    tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
    model = AutoModelForCausalLM.from_pretrained(model_save_directory).to("cuda")
    print(model)
    print("------------------------------------------------------")
    
    large_lm= transformers.pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer,
        # device_map="auto",
        device = "cuda", 
        )

    return large_lm,tokenizer


def load_t5(model_name):
    model_save_directory = os.path.join("../models",model_name) 

    tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
    if model_name in ["google/t5-small","google/flan-t5-small"]:
        model = T5ForConditionalGeneration.from_pretrained(model_save_directory).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_save_directory).to("cuda")
    print(model)
    print("---------------------------------------")
    
    return model,tokenizer


def fillter_rationale(large_lm_name,small_lm_name,dataset_name,dir_name,split='train',
                      inference_num=20 , wo_opinion_rate=0.2, step2=True):

    dataset = load_dataset("json",data_files=os.path.join("../data","raw",
                            dataset_name,"{}.json".format(split)))[split] #['train']
    print(dataset)
    num_items = len(dataset)

    '''
    Template
    Question: 
    What type of water formation is formed by clouds? 
    (A) pearls (B) streams (C) shells (D) diamonds (E) rain (F) beads (G) cooled (H) liquid.
    Opinion: 
    I think the answer is (F),what do you think about? Why?Please think step by step.
    Answer: The correct answer is
    '''
    
    log_save_directory = os.path.join("../log","{}.json".format(dir_name))
    
    dataset_acc = {"wo":0,"right":0,"wrong":0}
    vote_count = 0  # 判断投票得出的结果与数据集提供答案的一致性 
    doubtful_answer_count = 0 # 判断多少数据集答案可疑
    pass_rate_count = 0
    pass_step1_count = 0
    pass_step2_count = 0
    
    if os.path.exists(log_save_directory):
        fl = open(log_save_directory)
        log = json.load(fl)
        dataset_acc["wo"] = log["acc_wo_count"]
        dataset_acc["right"] = log["acc_right_count"]
        dataset_acc["wrong"] = log["acc_wrong_count"]
        vote_count = log["voting_answer_count"]
        pass_rate_count = log["passing_rate_count"]
    else:
        fl = open(log_save_directory,mode="w")

    processed_data_save_directory = os.path.join("../data","processed","{}.jsonl".format(dir_name))
    
    if os.path.exists(processed_data_save_directory):
        i = len(open(processed_data_save_directory).readlines())
    else:
        i = 0
    print(i)
    f = open(processed_data_save_directory,mode="a")

    large_lm,large_tokenizer = load_llama(model_name=large_lm_name)
    small_lm,small_tokenizer = load_t5(model_name=small_lm_name)
    
    # i是当前问题的index
    while i < len(dataset):
        item = dataset[i]
        i += 1
        question = item['formatted_question']
        print("=================================================================")
        print("Question {}: ".format(i)+question)

        # Step 1 ------------------------

        print("Generate and Prefilter rationales. ----------------------------------------")
        num_of_choice = len(item['choices']['label']) #需注意不同数据集标签名不同
        
        ground_answer,voting_answer,pre_filter_rationales,item_acc,sycophancy = generate_answer(
                        large_lm=large_lm,tokenizer=large_tokenizer, 
                        question=question, num_of_choice=num_of_choice,
                        wo_opinion_rate=wo_opinion_rate, inference_num=inference_num,
                        has_answer=item['answerKey'])
        # print("({}) is the ground answer of question 【{}】!".format(ground_answer,question))

        if 'answerKey' in item and item['answerKey'] != "":
            if voting_answer == item['answerKey']:
                vote_count += 1
            print("Temporary ACC of self-consistency is {}".format(vote_count/i))
            if ground_answer != item['answerKey']:
                doubtful_answer_count += 1
            print("There are {} doubtful dataset answer.".format(doubtful_answer_count))

        for key in dataset_acc.keys():
            if key in item_acc.keys():
                dataset_acc[key] += item_acc[key]
            else:
                dataset_acc[key] == 0
        if wo_opinion_rate == 1.0:
            print("Temporary ACC is : wo-{}".format(dataset_acc["wo"]/i))
        else:
            print("Temporary ACC is : wo-{},right-{},wrong-{}".format(dataset_acc["wo"]/i,dataset_acc["right"]/i,dataset_acc["wrong"]/i))

        answer_text = item['choices']['text'][ord(ground_answer)-ord('A')]
        answer = ["({})".format(ground_answer),answer_text]
        print("Ground answer: {}".format(answer))

        # print(pre_filter_rationales)
        pass_step1_count += len(pre_filter_rationales)
        print("Generate {} rationale(s) leading to the correct answer.".format(len(pre_filter_rationales)))
        
        # Step 2 ------------------------
        
        if step2:

            print("Select golden rationales. -------------------------------------")

            golden_rationales = select_rationale(model=small_lm,tokenizer=small_tokenizer,
                            question = question,ground_answer = answer,
                            pre_filter_rationales = pre_filter_rationales)
            # print(golden_rationales)
            pass_step2_count += len(pre_filter_rationales)
            print("After selection,there are {} rationale(s) left.".format(len(golden_rationales)))
            pass_rate = len(golden_rationales)/len(pre_filter_rationales)
            pass_rate_count += pass_rate
            print("Passing rate: {},temporary average passing rate is {}".format(pass_rate,pass_rate_count/i))

        else:
            golden_rationales = pre_filter_rationales

        if len(golden_rationales) == 0:
            continue

        # 存储 question(choices)+ answer+ golden rationales
        # 写入路径： ../data/processed/qasc/
        # 格式：json lines
        # 效果：因数据集完整跑下来耗时较长，应可以记录跑了多少，再启动时接着上次的跑
        write_in = {"Question":question,
                    "Num of choice":num_of_choice,
                    "Answer":ground_answer,
                    "Rationales":golden_rationales}
        f.write(json.dumps(write_in, ensure_ascii=False) + "\n")

        fl = open(log_save_directory,mode="w")
        log_in = {"acc_wo_count":dataset_acc["wo"],
                  "acc_right_count":dataset_acc["right"],
                  "acc_wrong_count":dataset_acc["wrong"],
                  "voting_answer_count":vote_count,
                  "passing_rate_count":pass_rate_count}
        fl.write(json.dumps(log_in,ensure_ascii=False))
        
    
    for key in dataset_acc.keys():
        dataset_acc[key] /= num_items
    print(dataset_acc)
    f.close()


def generate_ft_data(dataset_name, use_opinion_ft, dir_name,sycophancy = None):
    print("Dataset:{}".format(dataset_name))
    in_dir = "../data/processed/{}.jsonl".format(dir_name)
    out_dir = "../data/finetuning/{}.jsonl".format(dir_name)
    fin = open(in_dir,mode="r+")
    fout = open(out_dir,mode="w+")

    for line in fin:
        item = json.loads(line)
        golden_rationales = item["Rationales"]
        num_of_rationales = len(golden_rationales)
        num_of_choice = item["Num of choice"]

        if not use_opinion_ft:
            for rationale in golden_rationales:
                write_in = {"Question":item["Question"],
                            "Answer":item["Answer"],
                            "Rationale":rationale}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
        elif not sycophancy:
            if num_of_choice + 1 >= num_of_rationales:
                # w/o opinion
                write_in = {"Question":item["Question"],
                            "Answer":item["Answer"],
                            "Rationale":golden_rationales[0]}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
                # with opinoin
                for i in range(1,num_of_choice+1):
                    opinion = chr(ord('A')+i-1)
                    if i < num_of_rationales:
                        rationale = golden_rationales[i]
                    else:
                        rationale = random.choice(golden_rationales)
                    write_in = {"Question":item["Question"],
                                "Opinion":opinion,
                                "Answer":item["Answer"],
                                "Rationale":rationale}
                    fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
            else:
                generate_time = num_of_rationales // (num_of_choice+1) + 1
                # w/o opinion
                for t in range(generate_time):
                    write_in = {"Question":item["Question"],
                                "Answer":item["Answer"],
                                "Rationale":golden_rationales[t]}
                    fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
                for i in range(1,num_of_choice+1):
                    opinion = chr(ord('A')+i-1)
                    for t in range(generate_time):
                        if generate_time*i + t < num_of_rationales:
                            rationale = golden_rationales[generate_time*i+t]
                        else:
                            rationale = random.choice(golden_rationales)
                        write_in = {"Question":item["Question"],
                                    "Opinion":opinion,
                                    "Answer":item["Answer"],
                                    "Rationale":rationale}
                        fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
        else: #针对易发生谄媚的opinion生成更多的fine-tune数据，但也需要生成无意见数据
            no_opinion_point = max(sycophancy.values()) // 2 + 1
            each_point_num = num_of_rationales // sum(sycophancy.values())
            # w/o opinion
            for t in range(no_opinion_point*each_point_num): 
                write_in = {"Question":item["Question"],
                            "Answer":item["Answer"],
                            "Rationale":random.choice(golden_rationales)}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
            for opinion,v in sycophancy:
                for t in range(v*each_point_num):
                    write_in = {"Question":item["Question"],
                                "Opinion":opinion,
                                "Answer":item["Answer"],
                                "Rationale":random.choice(golden_rationales)}
                    fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
    
    fin.close()
    fout.close()
    

def dataset_download(dataset_name):
    # 下载数据集到本地
    dataset = load_dataset(dataset_name)
    print(dataset)
    for split,ds in dataset.items():
        dataset_save_directory = os.path.join("../data","raw",dataset_name,f"{split}.json")
        ds.to_json(dataset_save_directory)
    print("Download {} dataset successfully! ------------------------")

def model_download(model_name):
    # 下载模型到本地
    model_name = ""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(model)

    pt_save_directory = os.path.join("../models",model_name) 
    tokenizer.save_pretrained(pt_save_directory) 
    model.save_pretrained(pt_save_directory)



if __name__ == "__main__":
    args = parse_args()
    
    print(args)
    large_lm = args.large_model
    small_lm = args.small_model
    dataset_name = args.dataset
    if args.dir_name is not None:
        dir_name = args.dir_name
    else:
        dir_name = dataset_name
        
    fillter_rationale(large_lm_name=large_lm, small_lm_name=small_lm,
                      dataset_name=dataset_name,dir_name=dir_name,step2=args.step2,
                      inference_num=args.inference_num,wo_opinion_rate=args.wo_opinion_rate)

    generate_ft_data(dataset_name,dir_name=dir_name,use_opinion_ft=args.use_opinion_ft)