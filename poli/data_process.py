import torch
import transformers
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM,T5ForConditionalGeneration
from datasets import load_dataset
import os.path
import re
from pre_filter import generate_answer,using_qa_generate_rationale,using_qa_generate_ar
from refined_selection import select_rationale,statistic_las,group_by_leaked
from datasets_load import datasets_load,load_preprocessed_data,load_finetuning_data,load_unprocessed_data,merge_dataset,subtract_dataset
import json
import random
import argparse
from tqdm import tqdm


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
    parser.add_argument(
        "--qa2r",
        type=int,
        default=0,
        help="When there is no rationale left,whether using QA to generate rationale."
    )
    return parser.parse_args()


def model_download(model_name):
    # 下载模型到本地

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "t5" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    print(model)

    pt_save_directory = os.path.join("../models",model_name) 
    tokenizer.save_pretrained(pt_save_directory) 
    model.save_pretrained(pt_save_directory)


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
    model = T5ForConditionalGeneration.from_pretrained(model_save_directory).to("cuda")
    
    print(model)
    print("---------------------------------------")
    
    return model,tokenizer


def fillter_rationale(large_lm_name,small_lm_name,dataset_name,dir_name,split='train',
                      inference_num=20 , wo_opinion_rate=0.2, step2=True,qa2r = 0):

    dataset = datasets_load(dataset_name,split)
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
    pass_step1_count = 0
    pass_step2_count = 0
    
    if os.path.exists(log_save_directory):
        fl = open(log_save_directory)
        log = json.load(fl)
        dataset_acc["wo"] = log["acc_wo_count"]
        dataset_acc["right"] = log["acc_right_count"]
        dataset_acc["wrong"] = log["acc_wrong_count"]
        vote_count = log["voting_answer_count"]
        pass_step1_count = log["pass_step1_count"]
        pass_step2_count = log["pass_step2_count"] # 留下的是通过两步筛选的
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
    
    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

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
            pass_step2_count += len(golden_rationales)
            print("After selection,there are {} rationale(s) left.".format(len(golden_rationales)))

            if pass_step1_count:
                pass_step2_rate = pass_step2_count/pass_step1_count
            else:
                pass_step2_rate = 0.0
            print("Temporary average passing rate is {}".format(pass_step2_rate))

        else:
            golden_rationales = pre_filter_rationales

        pbar.update(1)

        if len(golden_rationales) == 0:
            if not qa2r:
                continue
            else:
                print(f"Using QA Generating {qa2r} rationale(s). ----------------------------------------")
                golden_rationales = using_qa_generate_rationale(
                                large_lm=large_lm, tokenizer= large_tokenizer,
                                question=question, answer=answer, generate_time=qa2r)

        # 存储 question(choices)+ answer+ golden rationales
        # 写入路径： ../data/processed/qasc/
        # 格式：json lines
        # 效果：因数据集完整跑下来耗时较长，应可以记录跑了多少，再启动时接着上次的跑
        write_in = {"Question":question,
                    "Num of choice":num_of_choice,
                    "Answer":answer,
                    "Rationales":golden_rationales}
        f.write(json.dumps(write_in, ensure_ascii=False) + "\n")

        fl = open(log_save_directory,mode="w")
        log_in = {"acc_wo_count":dataset_acc["wo"],
                  "acc_right_count":dataset_acc["right"],
                  "acc_wrong_count":dataset_acc["wrong"],
                  "voting_answer_count":vote_count,
                  "pass_step1_count":pass_step1_count,
                  "pass_step2_count":pass_step2_count}
        fl.write(json.dumps(log_in,ensure_ascii=False))
        
        
    pbar.close()
    for key in dataset_acc.keys():
        dataset_acc[key] /= num_items
    print(dataset_acc)
    f.close()


def generate_ft_data(dataset_name, dir_name, use_opinion_ft = False, sycophancy = None):
    print("Dataset:{}".format(dataset_name))
    in_dir = "../data/processed/{}.jsonl".format(dir_name)
    out_dir = "../data/finetuning/{}.jsonl".format(dir_name)
    fin = open(in_dir,mode="r+")
    fout = open(out_dir,mode="w+")

    for line in fin:
        item = json.loads(line)
        question = item['Question']
        golden_rationales = item["Rationales"]
        num_of_rationales = len(golden_rationales)
        num_of_choice = item["Num of choice"]
        
        answer = item['Answer']
        # 由于生成数据的代码版本不同，对数据进行规范化
        num_of_choice = item['Num of choice']
        if not isinstance(answer,list):
            if answer != chr(ord('A')+num_of_choice-1):
                pattern_str = f'\({answer}\)' + '.*?\('
                answer_text = re.search(pattern_str,question).group()[4:-2]
            else:
                pattern_str = f'\({answer}\).*'
                answer_text = re.search(pattern_str,question).group()[4:]
            answer = ["({})".format(answer),answer_text]

        if not use_opinion_ft: # 默认，只生成QAR
            for rationale in golden_rationales:
                write_in = {"Question":question,
                            "Answer":answer,
                            "Rationale":rationale}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
        elif not sycophancy:
            if num_of_choice + 1 >= num_of_rationales:
                # w/o opinion
                write_in = {"Question":question,
                            "Answer":answer,
                            "Rationale":golden_rationales[0]}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
                # with opinoin
                for i in range(1,num_of_choice+1):
                    opinion = chr(ord('A')+i-1)
                    if i < num_of_rationales:
                        rationale = golden_rationales[i]
                    else:
                        rationale = random.choice(golden_rationales)
                    write_in = {"Question":question,
                                "Opinion":opinion,
                                "Answer":answer,
                                "Rationale":rationale}
                    fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
            else:
                generate_time = num_of_rationales // (num_of_choice+1) + 1
                # w/o opinion
                for t in range(generate_time):
                    write_in = {"Question":question,
                                "Answer":answer,
                                "Rationale":golden_rationales[t]}
                    fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
                for i in range(1,num_of_choice+1):
                    opinion = chr(ord('A')+i-1)
                    for t in range(generate_time):
                        if generate_time*i + t < num_of_rationales:
                            rationale = golden_rationales[generate_time*i+t]
                        else:
                            rationale = random.choice(golden_rationales)
                        write_in = {"Question":question,
                                    "Opinion":opinion,
                                    "Answer":answer,
                                    "Rationale":rationale}
                        fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
        else: #针对易发生谄媚的opinion生成更多的fine-tune数据，但也需要生成无意见数据
            no_opinion_point = max(sycophancy.values()) // 2 + 1
            each_point_num = num_of_rationales // sum(sycophancy.values())
            # w/o opinion
            for t in range(no_opinion_point*each_point_num): 
                write_in = {"Question":question,
                            "Answer":answer,
                            "Rationale":random.choice(golden_rationales)}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
            for opinion,v in sycophancy:
                for t in range(v*each_point_num):
                    write_in = {"Question":question,
                                "Opinion":opinion,
                                "Answer":answer,
                                "Rationale":random.choice(golden_rationales)}
                    fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
    
    fin.close()
    fout.close()
    

def statistic_leakage_data(eval_model_name,dir_name,grouping=True):
    # 对泄露进行统计分析：LAS，分析对象是QAR数据集(重点在R)，不是model
    eval_model,tokenizer = load_t5(model_name=eval_model_name)
    dataset = load_finetuning_data(dir_name)

    # 将是否泄露分组统计
    if grouping: 
        leakage_rationales,no_leakage_rationales = group_by_leaked(eval_model,tokenizer,dataset)

        leaked_rate = len(leakage_rationales)/len(dataset)
        print(f"The leaked rate of dataset {dir_name} is {leaked_rate} .")

        leakage_result = statistic_las(eval_model,tokenizer,leakage_rationales)
        print("Leakage rationales performance of metrics: {}".format(leakage_result))

        no_leakage_result = statistic_las(eval_model,tokenizer,no_leakage_rationales)
        print("No leakage rationales performance of metrics: {}".format(no_leakage_result))

        las = (leakage_result['LAS'] + no_leakage_result['LAS'])/2
    else:
        result = statistic_las(eval_model,tokenizer,dataset)
        las = result['LAS']

    print("The LAS of dataset is {} .".format(las))
    

def step2_selection(dir_name,small_lm_name,output):
    
    dataset = load_preprocessed_data(dir_name)
    small_lm,small_tokenizer = load_t5(model_name=small_lm_name)
    output_dir = "../data/processed/{}.jsonl".format(output)
    fout = open(output_dir,mode="a+")

    # 通过率计算
    total = 0
    pass_count = 0

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    print("Select golden rationales. -------------------------------------")

    for item in dataset:

        question = item['Question']
        print(question)
        answer = item['Answer']
        # 由于生成数据的代码版本不同，对数据进行规范化
        num_of_choice = item['Num of choice']
        if not isinstance(answer,list):
            if answer != chr(ord('A')+num_of_choice-1):
                pattern_str = f'\({answer}\)' + '.*?\('
                answer_text = re.search(pattern_str,question).group()[4:-2]
            else:
                pattern_str = f'\({answer}\).*'
                answer_text = re.search(pattern_str,question).group()[4:]
            answer = ["({})".format(answer),answer_text]
        
        pre_filter_rationales = item['Rationales']

        golden_rationales = select_rationale(model=small_lm,tokenizer=small_tokenizer,
                            question = question,ground_answer = answer,
                            pre_filter_rationales = pre_filter_rationales)
        
        total += len(pre_filter_rationales)
        pass_count += len(golden_rationales)
        
        print("After selection,there are {} rationale(s) left.".format(len(golden_rationales)))        
        print("Temporary average passing rate is {}".format(pass_count/total))

        if len(golden_rationales):
            write_in = {"Question":question,
                        "Num of choice":num_of_choice,
                        "Answer":answer,
                        "Rationales":golden_rationales}
            fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")

        pbar.update(1)
    
    pbar.close()
    fout.close()


def STaR(model_name,dataset_name,dir_name="self_consistency1",generate_time=20):
    # 作为baseline，此实验在self-consistency的基础上进行
    # 对未回答正确的，利用 Q+A（Hint）生成 A+R，并对A进行filter，A正确时对应的的R保留
    model,tokenizer = load_llama(model_name=model_name)
    dataset = load_unprocessed_data(dataset_name,dir_name)
    out_dir = "../data/other/QA2RA.jsonl"
    fout = open(out_dir,mode="a+")

    pbar = tqdm(total=len(dataset))
    pbar.set_description("STaR...")

    for item in dataset:
        
        question = item['formatted_question']
        answer_key = item['answerKey']
        answer_text = item['choices']['text'][ord(answer_key)-ord('A')]
        answer = ["({})".format(answer_key),answer_text]
        num_of_choice = len(item['choices']['label'])

        rationales = using_qa_generate_ar(model,tokenizer,question,answer,generate_time)
        pbar.update(1)

        if len(rationales) == 0:
            continue
    
        write_in = {"Question":question,
                    "Num of choice":num_of_choice,
                    "Answer":answer,
                    "Rationales":rationales}
        fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")    
    
    pbar.close()
    fout.close()

    # 合并之前处理的数据与此次处理的数据到STaR.jsonl文件，并生成fine-tune数据
    merge_dataset(dir1="../data/processed/self_consistency1.jsonl",
                  dir2=out_dir,merged_dir="../data/processed/STaR.jsonl")
    generate_ft_data("qasc","STaR")


def duplicate_hard_question_rationales(dataset_name,dir_name,duplicate_num = 4):
    # 在同一数据集上，step1得到了正确回答而self-consistency未得到的问题可视为hard
    # 对此类hard问题，在生成fine-tune数据时增大权重
    # 直接生成fine_tune格式数据
    # 需要注意的是，duplicate_num为1表示复制一次，即在最终数据中hard问题出现两次
    # 这是因为最后有合并的环节，合并对象包括了easy + hard
    print("Dataset:{}".format(dataset_name))

    dir1 = "../data/processed/step12_base.jsonl"
    dir2 = "../data/processed/self_consistency1.jsonl"
    out_dir = "../data/other/duplicated.jsonl"
    hard_data = subtract_dataset(dir1,dir2,filter_attribute="Question")

    fout = open(out_dir,mode="a+")

    duplicate_int = duplicate_num // 1
    duplicate_frac = duplicate_num % 1

    for i in range(duplicate_int):
        for item in hard_data:
            rationales = item['Rationales']
            for rationale in rationales:
                write_in = {"Question":item['Question'],
                            "Answer":item['Answer'],
                            "Rationale":rationale}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
    if duplicate_frac:
        sample_num = duplicate_frac * len(hard_data)
        for i in sample_num:
            sampled_item = random.choice(hard_data)
            rationales = sampled_item['Rationales']
            for rationale in rationales:
                write_in = {"Question":sampled_item['Question'],
                            "Answer":sampled_item['Answer'],
                            "Rationale":rationale}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
    
    fout.close()

    # 合并easy数据
    merge_dataset(dir1=out_dir,dir2="../data/finetuning/step12_base.jsonl",
                  merged_dir=dir_name)


def sample_rationales(dir_name, sample_rate=0.5):
    dataset = load_preprocessed_data(dir_name)
    out_dir = f"../data/processed/{dir_name}_{sample_rate}sample.jsonl"
    fout = open(out_dir,mode="a+")
    for item in dataset:
        rationales = item['Rationales']
        sample_num = int(sample_rate * len(rationales)) if len(rationales) > 1 else 1
        sampled = random.sample(rationales,sample_num)

        write_in = {"Question":item['Question'],
                    "Num of choice":item['Num of choice'],
                    "Answer":item['Answer'],
                    "Rationales":sampled}
        fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")

    fout.close() 



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
    
    # TODO： qa2r 的处理可以在产生数据的过程中顺带进行，也可单独读取文件再做处理


    fillter_rationale(large_lm_name=large_lm, small_lm_name=small_lm,
                      dataset_name=dataset_name,dir_name=dir_name,step2=args.step2,
                      inference_num=args.inference_num,wo_opinion_rate=args.wo_opinion_rate,
                      qa2r=args.qa2r)

    generate_ft_data(dataset_name,dir_name=dir_name,use_opinion_ft=args.use_opinion_ft)