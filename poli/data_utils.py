import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,T5ForConditionalGeneration
import os.path
import re
from data_process import load_llama,load_t5,generate_ft_data
from datasets_load import *
from pre_filter import using_hint_generate_ar,statistic_failed_ar
from refined_selection import group_by_leaked,statistic_las,q2a,qr2a
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns


def statistic_failed_generation(large_lm_name,small_lm_name,dataset_name):
    dataset = datasets_load(dataset_name,split='train')
    print(dataset)
    large_lm,large_tokenizer = load_llama(large_lm_name)
    small_lm,small_tokenizer = load_t5(small_lm_name)

    output_dir = os.path.join("../data/other/failed_ar_wo6.jsonl")
    fout = open(output_dir,mode="a+")

    print("Generate and Prefilter rationales. ----------------------------------------")

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")
    prob_lift_list = []
    for item in dataset:
        question = item['formatted_question']
        print("=================================================================")
        print(f"Question: {question}")

        answer_key = item['answerKey']
        answer_text = item['choices']['text'][ord(answer_key)-ord('A')]
        answer = ["({})".format(answer_key),answer_text]
        print(answer)

        # test w/o opinion
        failed_rationales,answers = statistic_failed_ar(large_lm,large_tokenizer,
                                            question,ground_answer=answer,generate_time=6)
        print(answers)
        print("Generate {} failed a-rs without opinion.".format(len(failed_rationales)))
        
        q2a_prob = q2a(small_lm,small_tokenizer,question,answer,prob=True)
        for rationale in failed_rationales:
            print(rationale)
            if rationale is None:
                rationale = ""
            qr2a_prob = qr2a(small_lm,small_tokenizer,question,answer,rationale,prob=True)
            prob_lift = qr2a_prob - q2a_prob
            prob_lift_list.append(prob_lift)
            print(f"The rationale lifted the {prob_lift} probility to answer correctly!")
        
        if len(failed_rationales):
            write_in = {"Question":question,
                        "Num of choice":len(item['choices']['label']),
                        "Answer":answer,
                        "Rationales":failed_rationales}
            fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        pbar.update(1)

    pbar.close()
    fout.close()
    print(f"Total #rationale FAILED: {len(prob_lift_list)}")
    
    # 统计 prob lift
    plt.figure(figsize=(10,8),dpi=80)
    sns.kdeplot(prob_lift_list,fill=True,color="#01a2d9",alpha=.7,cut=0,clip=(-1,1))
    plt.savefig("failed_ar_wo6.png")


def statistic_leakage_data(eval_model_name,dataset_name,dir_name,grouping=True):
    # 对泄露进行统计分析：LAS，分析对象是QAR数据集(重点在R)，不是model
    eval_model,tokenizer = load_t5(model_name=eval_model_name)
    dataset = load_finetuning_data(dataset_name,dir_name)

    # 将是否泄露分组统计
    if grouping: 
        leakage_rationales,no_leakage_rationales = group_by_leaked(eval_model,tokenizer,dataset)

        leaked_rate = len(leakage_rationales)/len(dataset)
        print(f"The leaked rate of dataset {dataset_name}({dir_name}) is {leaked_rate} .")

        leakage_result = statistic_las(eval_model,tokenizer,leakage_rationales)
        print("Leakage rationales performance of metrics: {}".format(leakage_result))

        no_leakage_result = statistic_las(eval_model,tokenizer,no_leakage_rationales)
        print("No leakage rationales performance of metrics: {}".format(no_leakage_result))

        las = (leakage_result['LAS'] + no_leakage_result['LAS'])/2
    else:
        result = statistic_las(eval_model,tokenizer,dataset)
        las = result['LAS']

    print("The LAS of dataset is {} .".format(las))


def set_question_difficulty_level(dataset_name,loop_count):
    # TODO：对问题进行难度评级，输出到 data/.../loop_{loop_count}文件中，相比raw文件多一个难度的属性
    # easy: 上一轮中without opinoin即回答正确的问题
    # hard: 上一轮中提供right opinion才回答正确的问题
    # very hard： 上一轮中均没有回答正确的问题
    pass
    

def duplicate_hard_question_rationales(dataset_name,dir_name,duplicate_num = 4):
    # 在同一数据集上，step1得到了正确回答而self-consistency未得到的问题可视为hard
    # 对此类hard问题，在生成fine-tune数据时增大权重
    # 直接生成fine_tune格式数据
    # 需要注意的是，duplicate_num为1表示复制一次，即在最终数据中hard问题出现两次
    # 这是因为最后有合并的环节，合并对象包括了easy + hard
    print("Dataset:{}".format(dataset_name))

    # dir1 = "../data/processed/step12_base.jsonl"
    dir1 = os.path.join("../data/processed",dataset_name,"step12_base.jsonl")
    # dir2 = "../data/processed/self_consistency1.jsonl"
    dir2 = os.path.join("../data/processed",dataset_name,"self_consistency1.jsonl")
    out_dir = "../data/other/{}_duplicated.jsonl".format(dataset_name)
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
    merge_dataset(dir1=out_dir,dir2=os.path.join("../data/finetuning/",dataset_name,"step12_base.jsonl"),
                  merged_dir=dir_name)


def STaR(model_name,dataset_name,dir_name="self_consistency1",generate_time=20):
    # 作为baseline，此实验在self-consistency的基础上进行
    # 对未回答正确的，利用 Q+A（Hint）生成 A+R，并对A进行filter，A正确时对应的的R保留
    model,tokenizer = load_llama(model_name=model_name)
    dataset = load_unprocessed_data(dataset_name,dir_name)
    out_dir = "../data/other/{}_QA2RA.jsonl".format(dataset_name)
    fout = open(out_dir,mode="a+")

    pbar = tqdm(total=len(dataset))
    pbar.set_description("STaR...")

    for item in dataset:
        
        question = item['formatted_question']
        answer_key = item['answerKey']
        answer_text = item['choices']['text'][ord(answer_key)-ord('A')]
        answer = ["({})".format(answer_key),answer_text]
        num_of_choice = len(item['choices']['label'])

        rationales = using_hint_generate_ar(model,tokenizer,question,answer,generate_time)
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


def sample_rationales(dataset_name, dir_name, sample_rate=0.5):
    dataset = load_preprocessed_data(dataset_name,dir_name)
    out_dir = os.path.join("../data/processed",dataset_name,f"{dir_name}_{sample_rate}sample.jsonl")
    fout = open(out_dir,mode="a+")
    for item in dataset:
        rationales = item['Rationales']

        # sample_num = int(sample_rate * len(rationales)) if len(rationales) > 1 else 1
        # sampled = random.sample(rationales,sample_num)
        sampled = []
        for rationale in rationales:
            if random.random() < sample_rate:
                sampled.append(rationale)

        if len(sampled) == 0:
            continue

        write_in = {"Question":item['Question'],
                    "Num of choice":item['Num of choice'],
                    "Answer":item['Answer'],
                    "Rationales":sampled}
        fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")

    fout.close() 


def filter_threshold(dataset_name, dir_name, threshold=-0.5):
    # filter the rationale by reward threshold
    out_dir = os.path.join("../data/processed",dataset_name,f"{dir_name}_{threshold}filter.jsonl")
    fout = open(out_dir,mode="a+")
    dir_path = os.path.join("../data/processed",dataset_name,f"{dir_name}.jsonl")

    file = open(dir_path,'r',encoding='utf-8')
    
    for line in file.readlines():
        item = json.loads(line)
        rationales = item['Rationales']
        passed = []
        
        for rationale,reward in rationales:
            if reward >= threshold:
                passed.append(rationale)
        
        if len(passed) == 0:
            continue

        write_in = {"Question":item['Question'],
                    "Num of choice":item['Num of choice'],
                    "Answer":item['Answer'],
                    "Rationales":passed}
        fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")

    fout.close() 


def filter_threshold_failed(dataset_name, dir_name, threshold=0.5):
    # filter the rationale by reward threshold
    out_dir = os.path.join("../data/processed",dataset_name,f"{dir_name}_{threshold}filter.jsonl")
    fout = open(out_dir,mode="a+")
    dir_path = os.path.join("../data/processed",dataset_name,f"{dir_name}.jsonl")

    file = open(dir_path,'r',encoding='utf-8')
    
    for line in file.readlines():
        item = json.loads(line)
        rationales = item['Rationales']
        passed = []
        
        for answer,rationale,reward in rationales:
            if reward >= threshold:
                passed.append([answer,rationale])
        
        if len(passed) == 0:
            continue

        write_in = {"Question":item['Question'],
                    "Num of choice":item['Num of choice'],
                    "True Answer":item['True Answer'],
                    "Rationales":passed}
        fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")

    fout.close() 


def select_best_worst(dataset_name, dir_name):
    best_dir = os.path.join("../data/finetuning",dataset_name,f"{dir_name}_best.jsonl")
    fbest = open(best_dir,mode="a+")
    worst_dir = os.path.join("../data/finetuning",dataset_name,f"{dir_name}_worst.jsonl")
    fworst = open(worst_dir,mode="a+")

    dir_path = os.path.join("../data/processed",dataset_name,f"{dir_name}.jsonl")
    file = open(dir_path,'r',encoding='utf-8')

    for line in file.readlines():
        item = json.loads(line)
        rationale_lists = item['Rationales']
        rationales,rewards = zip(*rationale_lists)
        best_rationale = rationales[rewards.index(max(rewards))]
        worst_rationale = rationales[rewards.index(min(rewards))]

        write_in = {"Question":item['Question'],
                    "Answer":item['Answer'],
                    "Rationale":best_rationale}
        fbest.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        write_in = {"Question":item['Question'],
                    "Answer":item['Answer'],
                    "Rationale":worst_rationale}
        fworst.write(json.dumps(write_in, ensure_ascii=False) + "\n")

    fbest.close()
    fworst.close()