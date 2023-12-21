import os
from datasets import load_dataset,concatenate_datasets,Dataset
import re
import json
import ast


def dataset_download(dataset_name,subset=None):
    # 下载数据集到本地
    if subset:
        dataset = load_dataset(dataset_name,subset)
        data_dir = os.path.join("../data/raw",dataset_name,subset)
    else:
        dataset = load_dataset(dataset_name)
        data_dir = os.path.join("../data/raw",dataset_name)
    print(dataset)
    for split,ds in dataset.items():
        dataset_save_directory = os.path.join(data_dir,f"{split}.json")
        ds.to_json(dataset_save_directory)
    print("Download {} dataset successfully! ------------------------".format(dataset_name))


def datasets_load(dataset_name,subset=None,split='train'):
    print(f"Loading {dataset_name} dataset. Subset:{subset} Split:{split} ----------------------------")
    # 需要的col：
    # formatted_question、choices（text+label）、answerKey
    if subset:
        dataset_dir = os.path.join("../data/raw",dataset_name,subset)
    else:
        dataset_dir = os.path.join("../data/raw",dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        dataset_download(dataset_name,subset)

    if dataset_name == "qasc":
        return load_qasc(split)
    elif dataset_name == "commonsense_qa":
        return load_commonsenseQA(split)
    elif dataset_name == "openbookqa":
        return load_openbookQA(split)
    elif dataset_name == "ai2_arc":
        return load_ai2arc(subset=subset,split=split)
    elif dataset_name == "aqua_rat":
        return load_aqua(split)
    # TODO：其他数据集之后往这里加


def math_datasets_load(dataset_name,subset=None,split='train'):
    # 无选项
    # 需要的col:
    # question、answerNum
    if subset:
        dataset_dir = os.path.join("../data/raw",dataset_name,subset)
    else:
        dataset_dir = os.path.join("../data/raw",dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        dataset_download(dataset_name,subset)
    
    if dataset_name == "gsm8k":
        return load_gsm8k(subset,split)


def load_qasc(split='Train'):
    dataset_name = "qasc"
    dataset_dir = os.path.join("../data/raw",dataset_name,"{}.json".format(split))
    dataset = load_dataset("json",data_files=dataset_dir)['train']
    # features: ['id', 'question', 'choices', 'answerKey',
    #           'fact1', 'fact2', 'combinedfact', 'formatted_question']
    dataset = dataset.select_columns(['formatted_question','choices','answerKey'])
    print(dataset)
    return dataset


def load_commonsenseQA(split='Train'):
    dataset_name = "commonsense_qa"
    dataset_dir = os.path.join("../data/raw",dataset_name,"{}.json".format(split))
    dataset = load_dataset("json",data_files=dataset_dir)['train']
    # features:['id','question','question_concept','choices','answerKey']
    dataset = dataset.map(get_formatted_question)
    dataset = dataset.select_columns(['formatted_question','choices','answerKey'])
    print(dataset)
    return dataset


def load_openbookQA(split='Train'):
    dataset_name = "openbookqa"
    dataset_dir = os.path.join("../data/raw",dataset_name,"{}.json".format(split))
    dataset = load_dataset("json",data_files=dataset_dir)['train']
    # features:['id','question_stem','choices','answerKey']
    dataset = dataset.rename_column('question_stem','question')
    dataset = dataset.map(get_formatted_question)
    dataset = dataset.select_columns(['formatted_question','choices','answerKey'])
    print(dataset)
    return dataset


def load_ai2arc(subset,split='Train'):
    dataset_name = "ai2_arc"
    dataset_dir = os.path.join("../data/raw",dataset_name,subset,"{}.json".format(split))
    dataset = load_dataset("json",data_files=dataset_dir)['train']
    # features:['id','question','choices','answerKey']
    dataset = dataset.map(get_formatted_question)
    dataset = dataset.select_columns(['formatted_question','choices','answerKey'])
    print(dataset)
    return dataset


def load_aqua(split='Train'):
    dataset_name = "aqua_rat"
    dataset_dir = os.path.join("../data/raw",dataset_name,"{}.json".format(split))
    dataset = load_dataset("json",data_files=dataset_dir)['train']
    # features:['question','options','rationale','correct']
    dataset = dataset.rename_column('correct','answerKey')
    def _process_data(item):
        question_stem = item['question']
        choices = {'label':[],'text':[]}
        options_str = ""
        for option in item['options']:
            split_list = option.split(")")
            key = split_list[0]
            text = split_list[-1]
            choices['label'].append(key)
            choices['text'].append(text)
            options_str += " ("+option
        item['formatted_question'] = question_stem + options_str
        item['choices'] = choices
        return item
    
    dataset = dataset.map(_process_data)
    dataset = dataset.select_columns(['formatted_question','choices','answerKey'])
    print(dataset)
    return dataset


def get_formatted_question(item):
    formatted_question = item['question']
    labels = item['choices']['label']
    num_of_choices = len(labels)
    texts = item['choices']['text']
    for i in range(num_of_choices):
        choice = f"({labels[i]})" + " " + texts[i]
        formatted_question += " " + choice
    item['formatted_question'] = formatted_question
    return item
    

def load_gsm8k(subset='main',split='Train'):
    dataset_name = "gsm8k"
    dataset_dir = os.path.join("../data/raw",dataset_name,subset,"{}.json".format(split))
    dataset = load_dataset("json",data_files=dataset_dir)['train']
    # features:['question','answer']
    def _process_data(item):
        rationale,answer = item['answer'].split("#### ")
        item['answerNum'] = answer.replace(",","") # 去掉分位符
        return item
    dataset = dataset.map(_process_data)
    dataset = dataset.select_columns(['question','answerNum'])
    print(dataset)
    return dataset


def load_preprocessed_data(dataset_name,dir_name):
    # cols:'Question', 'Num of choice', 'Answer', 'Rationales'(list)
    dataset_dir = os.path.join("../data/processed",dataset_name,f"{dir_name}.jsonl")
    try:
        dataset = load_dataset('json',data_files=dataset_dir)['train']
        print(dataset)
    except:
        dataset = load_unformatted_data(dataset_dir)
    return dataset


def load_finetuning_data(dataset_name,dir_name):
    # cols:'Question','Answer','Rationale'(,'Opinion')
    dataset_dir = os.path.join("../data/finetuning",dataset_name,f"{dir_name}.jsonl")
    dataset = load_dataset('json',data_files=dataset_dir)['train']
    print(dataset)
    return dataset


def load_ppo_data(dataset_name,dir_name):
    # cols:'Query','Response','Reward'
    dataset_dir = os.path.join("../data/ppo",dataset_name,f"{dir_name}.jsonl")
    dataset = load_dataset('json',data_files=dataset_dir)['train']
    print(dataset)
    return dataset


def load_dpo_data(dataset_name,dir_name):
    # cols:'prompt','chosen','rejected',('score')
    dataset_dir = os.path.join("../data/dpo",dataset_name,f"{dir_name}.jsonl")
    dataset = load_dataset('json',data_files=dataset_dir)['train']
    print(dataset)
    return dataset


def load_unformatted_data(dir_path):
    # 应对执行load_dataset时出错的情形，逐行读取
    # 出错主要在于rationale项混杂了str（rationale）和float（reward），无法生成dataset
    # 处理时将reward单独归为一项生成dataset
    file = open(dir_path,'r',encoding='utf-8')
    datas = []
    for line in file.readlines():
        dic = json.loads(line)
        rationales = []
        rewards = []
        for rationale,reward in dic['Rationales']:
            rationales.append(rationale)
            rewards.append(reward)
        dic['Rationales'] = rationales
        dic['Rewards'] = rewards
        datas.append(dic)
    dataset = Dataset.from_list(datas)
    print(dataset)
    return dataset


def load_unprocessed_data(dataset_name,dir_name):
    # 对于某数据集，提取出未经处理的（被筛掉而未进入下一步）的剩余数据
    # 主要用于STaR
    print("Getting unprocessed question...")
    save_dir = f"../data/other/{dataset_name}_notIn_{dir_name}.jsonl"
    if not os.path.exists(save_dir):
        raw_dataset = datasets_load(dataset_name)
        processed_data = load_preprocessed_data(dataset_name,dir_name)
        unprocessed_data = raw_dataset.filter(lambda x:x['formatted_question'] 
                                            not in processed_data['Question'])
        unprocessed_data.to_json(save_dir)
    else:
        unprocessed_data = load_dataset('json',data_files=save_dir)['train']
    print(unprocessed_data)
    return unprocessed_data


def merge_dataset(dir_list,merged_dir=None):
    # 合并数据集
    dataset_list = []
    for dir in dir_list:
        dataset = load_dataset('json',data_files=dir)['train']
        dataset_list.append(dataset)
    # 合并前注意对各属性格式进行统一，尤其Answer
    merged_data = concatenate_datasets(dataset_list)
    print(merged_data)
    if merged_dir:
        merged_data.to_json(merged_dir)
    return merged_data


def subtract_dataset(dir1,dir2,filter_attribute,subtracted_dir=None):
    # dir1的数据集 减 dir2的数据集 ,根据某一属性做差
    dataset1 = load_dataset('json',data_files=dir1)['train']
    dataset2 = load_dataset('json',data_files=dir2)['train']
    subtracted_data = dataset1.filter(lambda x:x[filter_attribute] 
                                    not in dataset2[filter_attribute])
    print(subtracted_data)
    if subtracted_dir:
        subtracted_data.to_json(subtracted_dir)
    return subtracted_data


def intersect_dataset(dir1,dir2,filter_attribute,intersect_dir=None):
    # 取dir1和dir2根据某一属性的 交集
    dataset1 = load_dataset('json',data_files=dir1)['train']
    dataset2 = load_dataset('json',data_files=dir2)['train']
    intersect_data = dataset1.filter(lambda x:x[filter_attribute] 
                                    in dataset2[filter_attribute])
    print(intersect_data)
    if intersect_dir:
        intersect_data.to_json(intersect_dir)
    return intersect_data


def format_answer(dataset_name,dir_name):
    # 处理后answer格式应为list，本函数对preprocessed文件夹下的数据进行规范化
    dataset = load_preprocessed_data(dataset_name,dir_name)
    out_dir = os.path.join("../data/processed",dataset_name,f"{dir_name}_formatted.jsonl")
    fout = open(out_dir,mode="a+")
    for item in dataset:
        question = item['Question']
        answer = item['Answer']
        num_of_choice = item['Num of choice']

        if not isinstance(answer,list):
            if answer != chr(ord('A')+num_of_choice-1):
                pattern_str = f'\({answer}\)' + '.*?\('
                answer_text = re.search(pattern_str,question).group()[4:-2]
            else:
                pattern_str = f'\({answer}\).*'
                answer_text = re.search(pattern_str,question).group()[4:]
            answer = ["({})".format(answer),answer_text]
        
        write_in = {"Question":question,
                        "Num of choice":num_of_choice,
                        "Answer":answer,
                        "Rationales":item['Rationales']}
        fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")
    
    fout.close()


def group_ft_data(dataset_name,dir_name,out_dir=None):
    # group（Q,A,R）ft data by question
    dataset = load_finetuning_data(dataset_name,dir_name)
    grouped_data = Dataset.from_dict({})
    
    for item in dataset:
        question = item['Question']
        answer = item['Answer']
        if len(grouped_data) > 0 and question in grouped_data['Question']:
            continue
        filter_items = grouped_data.filter(lambda x: x['Question']==question)
        rationales = []
        for same_question_item in filter_items:
            assert same_question_item['Answer']==answer ,"Answer must be the same!"
            rationales.append(same_question_item['Rationale'])
        write_in = {"Question":question,
                    "Num of choice":item["Num of choice"],
                    "Answer":answer,
                    "Rationales":rationales}
        grouped_data = grouped_data.add_item(write_in)
    
    print(grouped_data)
    if out_dir:
        grouped_data.to_json(os.path.join("../data/processed",dataset_name,f"{out_dir}.jsonl"))
    return grouped_data


def join_processed_dataset(dataset_name,dir1,dir2,joined_dir,outer_join=True):
    dataset1 = load_preprocessed_data(dataset_name,dir1)
    dataset2 = load_preprocessed_data(dataset_name,dir2)
    fout = open(os.path.join("../data/processed",dataset_name,f"{joined_dir}.jsonl"),mode="a+")
    has_reward = 'Rewards' in dataset1.column_names

    if outer_join:
        for item in dataset1:
            question = item["Question"]
            answer = item["Answer"]
            rationales = item["Rationales"]
            if has_reward:
                rewards = item['Rewards']
            same_question_item = dataset2.filter(lambda x:x["Question"]==question)
            if len(same_question_item):
                same_question_item = same_question_item[0]
                assert same_question_item['Answer']==answer ,"Answer must be the same!"
                rationales += same_question_item['Rationales']
                if has_reward:
                    rewards += same_question_item['Rewards']
            write_in = {"Question":question,
                        "Num of choice":item["Num of choice"],
                        "Answer":answer,
                        "Rationales":rationales}
            if has_reward:
                write_in['Rewards'] = rewards
            fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")
        for item in dataset2.filter(lambda x: x["Question"] not in dataset1["Question"]):
            fout.write(json.dumps(dict(item), ensure_ascii=False) + "\n")          
    else:    
        dataset1 = dataset1.filter(lambda x: x['Question'] in dataset2['Question'])
        dataset2 = dataset2.filter(lambda x: x['Question'] in dataset1['Question'])
        for item in dataset1:
            question = item["Question"]
            answer = item["Answer"]
            rationales = item["Rationales"]
            if has_reward:
                rewards = item['Rewards']
            same_question_item = dataset2.filter(lambda x:x["Question"]==question)[0]
            assert same_question_item['Answer']==answer ,"Answer must be the same!"
            rationales += same_question_item['Rationales']
            if has_reward:
                    rewards += same_question_item['Rewards']
            write_in = {"Question":question,
                        "Num of choice":item["Num of choice"],
                        "Answer":answer,
                        "Rationales":rationales}
            if has_reward:
                write_in['Rewards'] = rewards
            fout.write(json.dumps(write_in, ensure_ascii=False) + "\n")
    
    fout.close()


def load_inference_log(log_dir,inference_keys=[],out_dir=None):
    file = open(log_dir,mode="r")
    num_of_answerlist = len(inference_keys)
    log_dataset = Dataset.from_dict({})
    item = {}
    current_state = -2
    # state表示：-2为即将读取====，-1为即将读取问题，0为即将读取正确答案
    # n为即将读取 answerlist第n个

    def _is_list(text):
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj,list):
                return True
        except:
            pass
        return False

    for line in file:
        text = line.strip()
        if current_state == -2 and text.startswith("==============="):
            item = {}
            current_state = -1
        elif current_state == -1 and text.startswith("Question:"):
            item['Question'] = text.replace("Question: ","")
            current_state = 0
        elif current_state == 0 and _is_list(text):
            item['Answer'] = ast.literal_eval(text)
            current_state = 1
        elif current_state > 0 and _is_list(text):
            item[f'{inference_keys[current_state-1]} list'] = ast.literal_eval(text)
            current_state += 1
            if current_state == num_of_answerlist + 1:
                log_dataset = log_dataset.add_item(item)
                current_state = -2
        else:
            continue
    
    print(log_dataset)
    if out_dir:
        log_dataset.to_json(out_dir)
    return log_dataset


def load_dataset_by_line(dir_path):
    file = open(dir_path,'r',encoding='utf-8')
    datas = []
    for line in file.readlines():
        dic = json.loads(line)
        datas.append(dic)
    dataset = Dataset.from_list(datas)
    print(dataset)
    return dataset