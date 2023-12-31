import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,T5ForConditionalGeneration,BitsAndBytesConfig
import os.path
import re
from llama_utils import generate_answer,using_qa_generate_rationale,using_opinion_generate_ar, \
                        answer_question,answer_math_question,using_opinion_generate_math_ar, \
                        judge_attempted_answer_math,judge_attempted_rationale_math
from t5_utils import select_rationale,get_rationale_type,q2a,qr2a,q2a_math,qr2a_math
from datasets_load import datasets_load,load_preprocessed_data,math_datasets_load
import json
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


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


def load_llama(model_name,pipeline=True):
    
    model_save_directory = os.path.join("../models",model_name) 

    tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_save_directory).to("cuda")
    if pipeline:
        model = transformers.pipeline(
                    task="text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    # device_map="auto",
                    device = "cuda"
                )
        
    print(model)
    # model = model.half()
    print(model.device)
    print(model.dtype)
    print("------------------------------------------------------")

    return model,tokenizer


def load_t5(model_name):
    model_save_directory = os.path.join("../models",model_name) 

    tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
    model = T5ForConditionalGeneration.from_pretrained(model_save_directory).to("cuda")
    
    print(model)
    print("---------------------------------------")
    
    return model,tokenizer


def fillter_rationale(large_lm_name,small_lm_name,dataset_name,dir_name,split='train',
                      inference_num=20 , wo_opinion_rate=0.2, step2=True,qa2r = 0):
    # 主要在无正确答案时进行，按比例分各个模块提问次数
    # 对每一个问题分别走流程（step1/sc -> step2 -> qa2r）
    
    dataset = datasets_load(dataset_name,split=split)
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
    
    log_save_directory = os.path.join("../log",dataset_name,"{}.json".format(dir_name))
    
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

    processed_data_save_directory = os.path.join("../data","processed",dataset_name,"{}.jsonl".format(dir_name))
    
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
    os.makedirs("../data/finetuning/{}".format(dataset_name),exist_ok=True)

    in_dir = os.path.join("../data/processed",dataset_name,"{}.jsonl".format(dir_name))
    out_dir = os.path.join("../data/finetuning",dataset_name,"{}.jsonl".format(dir_name))

    fin = open(in_dir,mode="r+")
    fout = open(out_dir,mode="w+")
    fout.truncate()

    for line in fin:
        item = json.loads(line)
        question = item['Question']
        golden_rationales = item["Rationales"]
        num_of_rationales = len(golden_rationales)
        
        answer = item['Answer']
        # 由于生成数据的代码版本不同，对数据进行规范化   
        if not isinstance(answer,list):
            num_of_choice = item['Num of choice']
            if answer != chr(ord('A')+num_of_choice-1):
                pattern_str = f'\({answer}\)' + '.*?\('
                answer_text = re.search(pattern_str,question).group()[4:-2]
            else:
                pattern_str = f'\({answer}\).*'
                answer_text = re.search(pattern_str,question).group()[4:]
            answer = ["({})".format(answer),answer_text]

        if not use_opinion_ft: # 默认，只生成QAR
            for rationale in golden_rationales:
                # rationale 为列表
                if isinstance(rationale,list):
                    rationale = rationale[0]
                write_in = {"Question":question,
                            "Answer":answer,
                            "Rationale":rationale}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
        elif not sycophancy:
            num_of_choice = item['Num of choice']
            if num_of_choice + 1 >= num_of_rationales:
                # w/o opinion
                write_in = {"Question":question,
                            "Answer":answer,
                            "Rationale":golden_rationales[0]}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
                # with opinion
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
    

def generate_ft_data_math(dataset_name, dir_name):
    print("Dataset:{}".format(dataset_name))
    os.makedirs("../data/finetuning/{}".format(dataset_name),exist_ok=True)

    in_dir = os.path.join("../data/processed",dataset_name,"{}.jsonl".format(dir_name))
    out_dir = os.path.join("../data/finetuning",dataset_name,"{}.jsonl".format(dir_name))

    fin = open(in_dir,mode="r+")
    fout = open(out_dir,mode="w+")
    fout.truncate()

    for line in fin:
        item = json.loads(line)
        question = item['Question']
        golden_rationales = item["Rationales"]
        answer = item['Answer']

        for rationale in golden_rationales:
            write_in = {"Question":question,
                        "Answer":answer,
                        "Rationale":rationale}
            fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
    
    fin.close()
    fout.close()


def generate_ppo_data(dataset_name, dir_name, reward_model_name):
    # 利用小模型为rationale打分
    # 输出到 data/.../ppo 文件夹中， 包含属性['query','response','reward']
    dataset = load_preprocessed_data(dataset_name,dir_name)
    out_dir = os.path.join("../data/ppo",dataset_name)
    os.makedirs(out_dir,exist_ok=True)
    fout = open(os.path.join(out_dir,f"{dir_name}.jsonl"),mode="a+")

    pbar = tqdm(total= len(dataset))
    pbar.set_description("Generate ppo data...")

    reward_model,tokenizer = load_t5(reward_model_name)

    count_reward = {-1:0, 0:0, 1:0, 2:0}

    QUERY_PROMPT = "Question: {Question}. What do you think the answer is? Why? \n Answer:"
    RESPONSE_PROMPT = "The correct answer is {Answer}. \n {Rationale}"

    for item in dataset:
        question = item['Question']
        answer = item['Answer']
        rationales = item['Rationales']
        print("--------------------------------")
        print(answer)

        for rationale in rationales:

            query = QUERY_PROMPT.format_map({'Question':question})
            response = RESPONSE_PROMPT.format_map({'Answer':" ".join(answer),'Rationale':rationale})
            reward = get_rationale_type(reward_model,tokenizer,question,answer,rationale)

            count_reward[reward] += 1
            write_in = {"query":query,
                        "response":response,
                        "Reward":reward}
            fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")

        pbar.update(1)
    
    pbar.close()
    fout.close()
    print(count_reward)
    

def generate_dpo_data_by_answer(dataset_name, chosen_dir, rejected_dir, out_dir):
    # 构建pair-wise data, 格式 prompt、chosen、rejected
    # 利用不同的answer+rationale
    chosen_dataset = load_preprocessed_data(dataset_name,chosen_dir)
    rejected_dataset = load_preprocessed_data(dataset_name,rejected_dir)
    os.makedirs(os.path.join("../data/dpo",dataset_name),exist_ok=True)
    fout = open(os.path.join("../data/dpo",dataset_name,f"{out_dir}.jsonl"),mode="a+")
    
    rejected_dataset = rejected_dataset.filter(lambda x: x['Question'] in chosen_dataset['Question'])
    chosen_dataset = chosen_dataset.filter(lambda x: x['Question'] in rejected_dataset['Question'])
    def _join_dataset(item):
        item['Chosen Rationales'] = item['Rationales']
        rejected_item = rejected_dataset.filter(lambda x:x['Question'] == item["Question"])[0]
        item['Rejected Rationales'] = rejected_item['Rationales']
        return item
    joined_dataset = chosen_dataset.map(_join_dataset)
    prob = isinstance(joined_dataset[0]['Rationales'][0],list)
    # ANSWER_PROMPT = "The correct answer is {}."
    ANSWER_PROMPT = "{}."
    QUESTION_PROMPT = "Question: {}.\nAnswer: The correct answer is "

    print(f"{len(joined_dataset)} Questions to generate dpo pairs.")
    print(joined_dataset)
    pbar = tqdm(total= len(joined_dataset))
    pbar.set_description("Generate dpo data...")

    for item in joined_dataset:
        question = item['Question']
        prompt = QUESTION_PROMPT.format(question)
        true_answer = item['Answer']
        if isinstance(true_answer,list):
            true_answer = " ".join(true_answer)
        chosen_rationales = item['Chosen Rationales']
        rejected_rationales = item['Rejected Rationales']

        if prob: # 根据prob值设计pair
            # 先根据reward对rationale排序
            chosen_rationales.sort(key=lambda x:x[-1],reverse=True) # 降序
            rejected_rationales.sort(key=lambda x:x[-1],reverse=False) # 升序
            chosen_num = len(chosen_rationales)
            rejected_num = len(rejected_rationales)
            for i in range(max(chosen_num,rejected_num)):
                if i < chosen_num:
                    chosen,chosen_score = chosen_rationales[i]
                else:
                    chosen,chosen_score = random.choice(chosen_rationales)
                chosen = ANSWER_PROMPT.format(true_answer) + chosen

                if i < rejected_num:
                    rejected_answer,rejected_rationale,rejected_score = rejected_rationales[i]
                else:
                    rejected_answer,rejected_rationale,rejected_score = random.choice(rejected_rationales)
                rejected = ANSWER_PROMPT.format(rejected_answer) + rejected_rationale

                write_in = {'prompt':prompt,
                            'chosen':chosen,
                            'rejected':rejected,
                            'score':chosen_score-rejected_score}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
        else: # 仅从chosen和rejected中各选一个组成pair
            chosen_num = len(chosen_rationales)
            rejected_num = len(rejected_rationales)
            for i in range(max(chosen_num,rejected_num)):
                if i < chosen_num:
                    chosen = chosen_rationales[i]
                else:
                    chosen = random.choice(chosen_rationales)
                chosen = ANSWER_PROMPT.format(true_answer) + chosen

                if i < rejected_num:
                    rejected_answer,rejected_rationale = rejected_rationales[i] 
                else:
                    rejected_answer,rejected_rationale = random.choice(rejected_rationales)
                rejected = ANSWER_PROMPT.format(rejected_answer) + rejected_rationale
            
                write_in = {'prompt':prompt,
                            'chosen':chosen,
                            'rejected':rejected}
                fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")

        pbar.update(1)    
    
    pbar.close()
    fout.close()


def generate_dpo_data_by_rationale(dataset_name, dir_name, gap=0.2, top=1):
    # 构建pair-wise data, 格式 prompt、chosen、rejected
    # 利用均回答正确，但rationale不同的data，必须带reward
    dataset = load_preprocessed_data(dataset_name,dir_name)
    assert 'Rewards' in dataset.column_names,"This function has to use rewards!"
    fout = open(os.path.join("../data/dpo",dataset_name,f"{dir_name}_gap{gap}.jsonl"),mode="a+")

    pbar = tqdm(total=len(dataset))
    pbar.set_description("Generate dpo data...")
    QUESTION_PROMPT = "Question: {Question}.\nAnswer: The correct answer is "
    ANSWER_PROMPT = "{Answer[0]} {Answer[1]}."

    count_question = 0
    for item in dataset:
        prompt = QUESTION_PROMPT.format_map(item)
        answer = ANSWER_PROMPT.format_map(item)
        rationales = item['Rationales']
        rewards = item['Rewards']
        assert len(rationales)==len(rewards), "The num of rationales and rewards must be the same!"
        
        rationales_with_rewards = [[rationales[i],rewards[i]] for i in range((len(rationales)))]
        rationales_with_rewards.sort(key=lambda x:x[1],reverse=True) # 根据rewards降序排序
        count_pair = 0
        for i in range(len(rationales)):
            for j in range(i+1,len(rationales)):
                if rationales_with_rewards[i][1] - rationales_with_rewards[j][1] >= gap:
                    chosen = rationales_with_rewards[i][0]
                    rejected = rationales_with_rewards[j][0]
                    write_in = {'prompt':prompt,
                                'chosen':answer + chosen,
                                'rejected':answer + rejected}
                    fout.write(json.dumps(write_in,ensure_ascii=False) + "\n")
                    count_pair += 1

        if count_pair:
            count_question += 1
        # print(f"Generate {count_pair} pairs using {len(rationales)} rationales.")
        pbar.update(1)
    
    print(f"{count_question} questions generated DPO pairs.")
    pbar.close()
    fout.close()


def step1_generate(large_lm_name,dataset_name,inference_num):
    # inference_num 应为包含 wo(no_opinion)、right(right_opinion)、wrong(wrong_opinion)三项的dict
    # 数据集包含正确答案
    # 这个函数会分类记录所有回答正确的rationale
    dataset = datasets_load(dataset_name,split='train')
    print(dataset)

    dir_name = "../data/processed/{}".format(dataset_name)
    os.makedirs(dir_name,exist_ok=True)
    if 'wo' in inference_num:
        fwo = open(os.path.join(dir_name,f"step1_wo{inference_num['wo']}.jsonl"),mode="a+")
        fwo_failed = open(os.path.join(dir_name,f"step1_wo{inference_num['wo']}_failed.jsonl"),mode="a+")
    if 'right' in inference_num:
        fright = open(os.path.join(dir_name,f"step1_right{inference_num['right']}.jsonl"),mode="a+")
        fright_failed = open(os.path.join(dir_name,f"step1_right{inference_num['right']}_failed.jsonl"),mode="a+")
    if 'wrong' in inference_num:
        fwrong = open(os.path.join(dir_name,f"step1_wrong{inference_num['wrong']}.jsonl"),mode="a+")
        fwrong_failed = open(os.path.join(dir_name,f"step1_wrong{inference_num['wrong']}_failed.jsonl"),mode="a+")

    pass_count = {"wo":0,"right":0,"wrong":0}    

    large_lm,tokenizer = load_llama(model_name=large_lm_name)
    print("Generate and Prefilter rationales. ----------------------------------------")

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    for item in dataset:
        question = item['formatted_question']
        print("=================================================================")
        print(f"Question: {question}")

        num_of_choice = len(item['choices']['label']) #需注意不同数据集标签名不同
        answer_key = item['answerKey']
        answer_text = item['choices']['text'][ord(answer_key)-ord('A')]
        answer = ["({})".format(answer_key),answer_text]
        print(answer)

        # without opinion
        if 'wo' in inference_num:
            wo_rationales,wo_answers,failed_wo_rationales = answer_question(large_lm,tokenizer,
                                question,ground_answer=answer,generate_time=inference_num['wo'])
            print(wo_answers)
            pass_count['wo'] += len(wo_rationales)
            print("Generate {} rationales without opinion.".format(len(wo_rationales)))
            print("TEMP PASSing RATE : {}".format(pass_count["wo"]/((pbar.n+1)*inference_num["wo"])))

            if len(wo_rationales):
                write_in = {"Question":question,
                            "Num of choice":num_of_choice,
                            "Answer":answer,
                            "Rationales":wo_rationales}
                fwo.write(json.dumps(write_in, ensure_ascii=False) + "\n")
            if len(failed_wo_rationales):
                write_in = {"Question":question,
                            "Num of choice":num_of_choice,
                            "True Answer":answer,
                            "Rationales":failed_wo_rationales}
                fwo_failed.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        # true opinion
        if 'right' in inference_num:
            right_rationales,right_answers,failed_right_rationales = using_opinion_generate_ar(large_lm,tokenizer,
                                question,opinion_choice=answer[0],ground_answer=answer,generate_time=inference_num['right'])
            print(right_answers)
            pass_count['right'] += len(right_rationales)
            print("Generate {} rationales using right opinion.".format(len(right_rationales)))
            print("TEMP PASSing RATE : {}".format(pass_count["right"]/((pbar.n+1)*inference_num["right"])))

            if len(right_rationales):
                write_in = {"Question":question,
                            "Num of choice":num_of_choice,
                            "Answer":answer,
                            "Rationales":right_rationales}
                fright.write(json.dumps(write_in, ensure_ascii=False) + "\n")
            if len(failed_right_rationales):
                write_in = {"Question":question,
                            "Num of choice":num_of_choice,
                            "True Answer":answer,
                            "Rationales":failed_right_rationales}
                fright_failed.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        # wrong opinion
        if 'wrong' in inference_num:
            wrong_rationales = []
            wrong_answers = []
            failed_wrong_rationales = []
            generate_time = inference_num['wrong'] // (num_of_choice-1)
            for i in range(num_of_choice):
                opinion_choice = chr(ord('A') + i)
                if opinion_choice == answer[0][1]:
                    continue
                rationales,answer_list,failed_rationales = using_opinion_generate_ar(
                            large_lm,tokenizer,question,opinion_choice=opinion_choice,
                            ground_answer=answer,generate_time=generate_time)
                wrong_rationales += rationales
                wrong_answers += answer_list
                failed_wrong_rationales += failed_rationales
            print(wrong_answers)
            pass_count['wrong'] += len(wrong_rationales)
            print("Generate {} rationales using wrong opinion.".format(len(wrong_rationales)))
            print("TEMP PASSing RATE : {}".format(pass_count["wrong"]/((pbar.n+1)*inference_num["wrong"])))

            if len(wrong_rationales):
                write_in = {"Question":question,
                            "Num of choice":num_of_choice,
                            "Answer":answer,
                            "Rationales":wrong_rationales}
                fwrong.write(json.dumps(write_in, ensure_ascii=False) + "\n")
            if len(failed_wrong_rationales):
                write_in = {"Question":question,
                            "Num of choice":num_of_choice,
                            "True Answer":answer,
                            "Rationales":failed_wrong_rationales}
                fwrong_failed.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        
        pbar.update(1)
    
    pbar.close()
    if 'wo' in inference_num:
        fwo.close()
        fwo_failed.close()
    if 'right' in inference_num:
        fright.close()
        fright_failed.close()
    if 'wrong' in inference_num:
        fwrong.close()
        fwrong_failed.close()

    print(pass_count)
    

def step1_generate_math(large_lm_name,dataset_name="gsm8k",inference_num=None):
    # 用于inference 数学应用题 数据集
    dataset = math_datasets_load(dataset_name,subset='main',split='train')
    print(dataset)

    dir_name = "../data/processed/{}".format(dataset_name)
    os.makedirs(dir_name,exist_ok=True)
    if 'wo' in inference_num:
        fwo = open(os.path.join(dir_name,f"step1_wo{inference_num['wo']}.jsonl"),mode="a+")
        fwo_failed = open(os.path.join(dir_name,f"step1_wo{inference_num['wo']}_failed.jsonl"),mode="a+")
    if 'right' in inference_num:
        fright = open(os.path.join(dir_name,f"step1_right{inference_num['right']}.jsonl"),mode="a+")
        fright_failed = open(os.path.join(dir_name,f"step1_right{inference_num['right']}_failed.jsonl"),mode="a+")

    pass_count = {"wo":0, "right":0}

    large_lm,tokenizer = load_llama(model_name=large_lm_name)
    print("Generate and Prefilter rationales. ----------------------------------------")

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    for item in dataset:
        question = item['question']
        answerNum = item['answerNum']
        print("=================================================================")
        print(f"Question: {question}")
        print(f"Correct answerNum: {answerNum}")

        # without opinion
        if 'wo' in inference_num:
            wo_rationales,wo_answers,failed_wo_rationales = answer_math_question(large_lm,tokenizer,
                        question,ground_answer=answerNum,generate_time=inference_num['wo'],n_shot=8)
            print(wo_answers)
            pass_count['wo'] += len(wo_rationales)
            print("Generate {} rationales without opinion.".format(len(wo_rationales)))
            print("TEMP PASSing RATE : {}".format(pass_count["wo"]/((pbar.n+1)*inference_num["wo"])))

            if len(wo_rationales):
                write_in = {"Question":question,
                            "Answer":answerNum,
                            "Rationales":wo_rationales}
                fwo.write(json.dumps(write_in, ensure_ascii=False) + "\n")
            if len(failed_wo_rationales):
                write_in = {"Question":question,
                            "True Answer":answerNum,
                            "Rationales":failed_wo_rationales}
                fwo_failed.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        # true opinion
        if 'right' in inference_num:
            right_rationales,right_answers,failed_right_rationales = using_opinion_generate_math_ar(large_lm,tokenizer,
                        question,opinion_num=answerNum,ground_answer=answerNum,generate_time=inference_num['right'],n_shot=8)
            print(right_answers)
            pass_count['right'] += len(right_rationales)
            print("Generate {} rationales using right opinion.".format(len(right_rationales)))
            print("TEMP PASSing RATE : {}".format(pass_count["right"]/((pbar.n+1)*inference_num["right"])))

            if len(right_rationales):
                write_in = {"Question":question,
                            "Answer":answerNum,
                            "Rationales":right_rationales}
                fright.write(json.dumps(write_in, ensure_ascii=False) + "\n")
            
            if len(failed_right_rationales):
                write_in = {"Question":question,
                            "True Answer":answerNum,
                            "Rationales":failed_right_rationales}
                fright_failed.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        
        pbar.update(1)
    
    pbar.close()
    if 'wo' in inference_num:
        fwo.close()
        fwo_failed.close()
    if 'right' in inference_num:
        fright.close()
        fright_failed.close()

    print(pass_count)


def step1_generate_math_wrong_opinion(large_lm_name,dataset_name,failed_dir,inference_num=10):
    # 利用wrong opinion inference
    # 错误的answerNum 来自failed inference answer
    dataset = load_preprocessed_data(dataset_name,failed_dir)
    print(dataset)

    fwrong = open(os.path.join("../data/processed",dataset_name,f"step1_wrong{inference_num}.jsonl"),mode="a+")
    fwrong_failed = open(os.path.join("../data/processed",dataset_name,f"step1_wrong{inference_num}_failed.jsonl"),mode="a+")
    pass_count = 0

    large_lm,tokenizer = load_llama(model_name=large_lm_name)
    print("Generate and Prefilter rationales. ----------------------------------------")

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    def _get_wrong_opinions(wrong_answers,inference_num):
        if len(wrong_answers) > inference_num:
            opinion_list = random.choices(wrong_answers,k=inference_num)
        else:
            num_of_choice = inference_num - len(wrong_answers)
            opinion_list =  wrong_answers + [random.choice(wrong_answers) for i in range(num_of_choice)]
        return dict(Counter(opinion_list))

    for item in dataset:
        question = item['Question']
        answerNum = item['True Answer']
        print("=================================================================")
        print(f"Question: {question}")
        print(f"Correct answerNum: {answerNum}")

        perturb_answers = [wrong_answer for wrong_answer,wrong_rationale in item['Rationales']]
        wrong_opinions = _get_wrong_opinions(perturb_answers,inference_num)
        print(wrong_opinions)

        wrong_rationales = []
        wrong_answers = []
        failed_wrong_rationales = []
        for opinion,generate_time in wrong_opinions.items():
            rationales,answer_list,failed_rationales = using_opinion_generate_math_ar(large_lm,tokenizer,
                question,opinion_num=opinion,ground_answer=answerNum,generate_time=generate_time,n_shot=8)
            wrong_rationales += rationales
            wrong_answers += answer_list
            failed_wrong_rationales += failed_rationales
        print(wrong_answers)
        pass_count += len(wrong_rationales)
        print("Generate {} rationales using wrong opinion.".format(len(wrong_rationales)))
        print("TEMP PASSing RATE : {}".format(pass_count/((pbar.n+1)*inference_num)))

        if len(wrong_rationales):
            write_in = {"Question":question,
                        "Answer":answerNum,
                        "Rationales":wrong_rationales}
            fwrong.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        if len(failed_wrong_rationales):
            write_in = {"Question":question,
                        "True Answer":answerNum,
                        "Rationales":failed_wrong_rationales}
            fwrong_failed.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        
        pbar.update(1)
    
    pbar.close()
    fwrong.close()
    fwrong_failed.close()


def step2_selection_simple_t5(dataset_name,dir_name,small_lm_name,output):
    # QR->A的即通过filter
    dataset = load_preprocessed_data(dataset_name,dir_name)
    small_lm,small_tokenizer = load_t5(model_name=small_lm_name)
    output_dir = os.path.join("../data/processed",dataset_name,"{}_simpleStep2.jsonl".format(output))
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


def step2_selection_t5(dataset_name,dir_name,small_lm_name,output,duplicate_num=2):
    # Q->A且QR->A的保留, Q->WA但QR->A的增加权重（默认翻倍）, Q->A但QR->WA的筛去
    dataset = load_preprocessed_data(dataset_name,dir_name)
    small_lm,small_tokenizer = load_t5(model_name=small_lm_name)
    output_dir = os.path.join("../data/processed",dataset_name,"{}.jsonl".format(output))
    fout = open(output_dir,mode="a+")

    # 通过率计算
    total = 0
    type_count = {-1:0, 0:0, 1:0, 2:0}

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    print("Select golden rationales. -------------------------------------")

    for item in dataset:

        question = item['Question']
        print(question)
        answer = item['Answer']
        print(answer)
        assert isinstance(answer,list),"The type of answer should be list."
        
        pre_filter_rationales = item['Rationales']
        total += len(pre_filter_rationales)
        golden_rationales = []

        for rationale in pre_filter_rationales:

            rationale_type = get_rationale_type(small_lm,small_tokenizer,question,answer,rationale)
            print(rationale_type,end=" ")
            type_count[rationale_type] += 1

            if rationale_type == -1:
                continue
            elif rationale_type == 0 or rationale_type == 1:
                golden_rationales.append(rationale)
            else: # rationale_type == 2
                for i in range(duplicate_num):
                    golden_rationales.append(rationale)

        print("After step2 selection,there are {} rationale(s) left.".format(len(golden_rationales)))

        if len(golden_rationales):
            write_in = {"Question":question,
                        "Num of choice":item['Num of choice'],
                        "Answer":answer,
                        "Rationales":golden_rationales}
            fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")

        pbar.update(1)
    
    pbar.close()
    fout.close()
    print(f"Total #rationale BEFORE: {total}")
    print("Type count:",end=" ")
    print(type_count)


def step2_selection_prob_t5(dataset_name,dir_name,small_lm_name,output,threshold=-1.0):
    # 通过Q->A QR->A 的logits计算得分，大于threshold的通过selection
    dataset = load_preprocessed_data(dataset_name,dir_name)
    small_lm,small_tokenizer = load_t5(model_name=small_lm_name)
    output_dir = os.path.join("../data/processed",dataset_name,"{}.jsonl".format(output))
    fout = open(output_dir,mode="a+")

    # 通过率计算
    total = 0
    pass_count = 0
    prob_lift_list = []

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    print("Select golden rationales. -------------------------------------")

    for item in dataset:

        question = item['Question']
        print(question)
        answer = item['Answer']
        print(answer)
        assert isinstance(answer,list),"The type of answer should be list."

        q2a_prob = q2a(small_lm,small_tokenizer,question,answer,prob=True)
        # print(q2a_prob)

        pre_filter_rationales = item['Rationales']
        total += len(pre_filter_rationales)
        golden_rationales = []

        for rationale in pre_filter_rationales:

            qr2a_prob = qr2a(small_lm,small_tokenizer,question,answer,rationale,prob=True)
            prob_lift = qr2a_prob - q2a_prob
            prob_lift_list.append(prob_lift)
            print(f"The rationale lifted the {prob_lift} probility to answer correctly!")

            if prob_lift >= threshold:
                # golden_rationales.append(rationale) 
                golden_rationales.append([rationale,prob_lift])
                pass_count += 1

        print("After step2 selection,there are {} rationale(s) left.".format(len(golden_rationales)))

        
        if len(golden_rationales):
            write_in = {"Question":question,
                        "Num of choice":item['Num of choice'],
                        "Answer":answer,
                        "Rationales":golden_rationales}
            fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        pbar.update(1)
    

    pbar.close()
    fout.close()
    print(f"Total #rationale BEFORE: {total}")
    print(f"Pass threshold rationale: {pass_count}")
    # print(f"All prob lift data: {prob_lift_list}")
    
    # 统计 prob lift
    plt.figure(figsize=(10,8),dpi=80)
    sns.kdeplot(prob_lift_list,fill=True,color="#01a2d9",alpha=.7,cut=0,clip=(-1,1))
    plt.savefig(output+".png")


def step2_selection_prob_math_t5(dataset_name,dir_name,small_lm_name,output,threshold=-1.0):
    # 通过Q->A QR->A 的logits计算得分，大于threshold的通过selection
    # 对于数学问题，不要求其回答正确数字，只要回答 yes/no 即可
    dataset = load_preprocessed_data(dataset_name,dir_name)
    small_lm,small_tokenizer = load_t5(model_name=small_lm_name)
    output_dir = os.path.join("../data/processed",dataset_name,"{}.jsonl".format(output))
    fout = open(output_dir,mode="a+")

    # 通过率计算
    total = 0
    pass_count = 0
    prob_lift_list = []

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    print("Select golden rationales. -------------------------------------")

    for item in dataset:

        question = item['Question']
        print(question)
        answerNum = item['Answer']
        print(f"True answerNum: {answerNum}")

        q2a_prob = q2a_math(small_lm,small_tokenizer,question,answerNum,prob=True)
        # print(q2a_prob)

        pre_filter_rationales = item['Rationales']
        total += len(pre_filter_rationales)
        golden_rationales = []

        for rationale in pre_filter_rationales:

            qr2a_prob = qr2a_math(small_lm,small_tokenizer,question,answerNum,rationale,prob=True)
            prob_lift = qr2a_prob - q2a_prob
            prob_lift_list.append(prob_lift)
            print(f"The rationale lifted the {prob_lift} probility to answer correctly!")

            if prob_lift >= threshold:
                # golden_rationales.append(rationale) 
                golden_rationales.append([rationale,prob_lift])
                pass_count += 1

        print("After step2 selection,there are {} rationale(s) left.".format(len(golden_rationales)))

        if len(golden_rationales):
            write_in = {"Question":question,
                        "Answer":answerNum,
                        "Rationales":golden_rationales}
            fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        pbar.update(1)
    

    pbar.close()
    fout.close()
    print(f"Total #rationale BEFORE: {total}")
    print(f"Pass threshold rationale: {pass_count}")
    # print(f"All prob lift data: {prob_lift_list}")
    
    # 统计 prob lift
    plt.figure(figsize=(10,8),dpi=80)
    sns.kdeplot(prob_lift_list,fill=True,color="#01a2d9",alpha=.7,cut=0,clip=(-1,1))
    plt.savefig(output+".png")


def step2_selection_prob_failed_t5(dataset_name,dir_name,small_lm_name,output,threshold=1.0):
    # 对回答失败数据进行处理，threshold为 上限
    dataset = load_preprocessed_data(dataset_name,dir_name)
    small_lm,small_tokenizer = load_t5(model_name=small_lm_name)
    output_dir = os.path.join("../data/processed",dataset_name,"{}.jsonl".format(output))
    fout = open(output_dir,mode="a+")

    # 通过率计算
    total = 0
    pass_count = 0
    prob_lift_list = []

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    print("Select golden rationales. -------------------------------------")

    for item in dataset:

        question = item['Question']
        print(question)
        true_answer = item['True Answer']
        print(true_answer)
        assert isinstance(true_answer,list),"The type of true answer should be list."

        q2a_prob = q2a(small_lm,small_tokenizer,question,true_answer,prob=True)
        # print(q2a_prob)

        pre_filter_rationales = item['Rationales']
        total += len(pre_filter_rationales)
        golden_rationales = []

        for answer,rationale in pre_filter_rationales:
            qr2a_prob = qr2a(small_lm,small_tokenizer,question,true_answer,rationale,prob=True)
            prob_lift = qr2a_prob - q2a_prob
            prob_lift_list.append(prob_lift)
            print(f"The rationale lifted the {prob_lift} probility to answer correctly!")

            if prob_lift <= threshold:
                # golden_rationales.append(rationale) 
                golden_rationales.append([answer,rationale,prob_lift])
                pass_count += 1

        print("After step2 selection,there are {} rationale(s) left.".format(len(golden_rationales)))
        
        if len(golden_rationales):
            write_in = {"Question":question,
                        "Num of choice":item['Num of choice'],
                        "True Answer":answer,
                        "Rationales":golden_rationales}
            fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        pbar.update(1)
    
    pbar.close()
    fout.close()
    print(f"Total #rationale BEFORE: {total}")
    print(f"Pass threshold rationale: {pass_count}")
    # print(f"All prob lift data: {prob_lift_list}")
    
    # 统计 prob lift
    plt.figure(figsize=(10,8),dpi=80)
    sns.kdeplot(prob_lift_list,fill=True,color="#01a2d9",alpha=.7,cut=0,clip=(-1,1))
    plt.savefig(output+".png")


def step2_selection_prob_math_llama(dataset_name,dir_name,large_lm_name,output):
    # 通过Q->A QR->A 的logits计算得分 ，不再使用QR2A-Q2A,直接用QR2A
    # 对于数学问题，不要求其回答正确数字，只要回答 yes/no 即可
    dataset = load_preprocessed_data(dataset_name,dir_name)
    dataset = dataset.select([2185+x for x in range(2000)])
    print(len(dataset))
    model,tokenizer = load_llama(model_name=large_lm_name,pipeline=False)
    output_dir = os.path.join("../data/processed",dataset_name,"{}.jsonl".format(output))
    fout = open(output_dir,mode="a+")

    total = 0
    rewards_list = []

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Processing data...")

    print("Select golden rationales. -------------------------------------")

    for item in dataset:

        question = item['Question']
        print(question)
        answerNum = item['Answer']
        print(f"True answerNum: {answerNum}")

        # q2a_prob = judge_attempted_answer_math(model,tokenizer,question,answerNum,prob=True)
        # print(q2a_prob)

        rationales = item['Rationales']
        total += len(rationales)
        rationales_with_rewards = []

        for rationale in rationales:
            
            qr2a_prob = judge_attempted_rationale_math(model,tokenizer,question,answerNum,rationale,prob=True)
            rewards_list.append(qr2a_prob)
            print(f"The rationale got score: {qr2a_prob}!")

            rationales_with_rewards.append([rationale,qr2a_prob])
        
        print(f"{len(rationales_with_rewards)} Rationales are scored.")
        write_in = {"Question":question,
                    "Answer":answerNum,
                    "Rationales":rationales_with_rewards}
        fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        pbar.update(1)
    

    pbar.close()
    fout.close()
    print(f"Total #rationale: {total}")
    # print(f"All prob lift data: {prob_lift_list}")
    
    # 统计 prob lift
    plt.figure(figsize=(10,8),dpi=80)
    sns.kdeplot(rewards_list,fill=True,color="#01a2d9",alpha=.7,cut=0,clip=(-1,1))
    plt.savefig(output+".png")


if __name__ == "__main__":
    args = parse_args()
    
    print(args)
    large_lm = args.large_model
    small_lm = args.small_model
    dataset_name = args.dataset
    if args.dir_name is not None:
        dir_name = args.dir_name
    else:
        dir_name = dataset_name + "simple"


    fillter_rationale(large_lm_name=large_lm, small_lm_name=small_lm,
                      dataset_name=dataset_name,dir_name=dir_name,step2=args.step2,
                      inference_num=args.inference_num,wo_opinion_rate=args.wo_opinion_rate,
                      qa2r=args.qa2r)

    generate_ft_data(dataset_name,dir_name=dir_name,use_opinion_ft=args.use_opinion_ft)