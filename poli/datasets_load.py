import os
from datasets import load_dataset,concatenate_datasets
import re
import json


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
    # TODO：其他数据集之后往这里加


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
    

def load_preprocessed_data(dataset_name,dir_name):
    # cols:'Question', 'Num of choice', 'Answer', 'Rationales'(list)
    dataset_dir = os.path.join("../data/processed",dataset_name,f"{dir_name}.jsonl")
    dataset = load_dataset('json',data_files=dataset_dir)['train']
    print(dataset)
    return dataset


def load_finetuning_data(dataset_name,dir_name):
    # cols:'Question','Answer','Rationale'(,'Opinion')
    dataset_dir = os.path.join("../data/finetuning",dataset_name,f"{dir_name}.jsonl")
    dataset = load_dataset('json',data_files=dataset_dir)['train']
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


def merge_dataset(dir1,dir2,merged_dir=None):
    # 合并数据集
    dataset1 = load_dataset('json',data_files=dir1)['train']
    dataset2 = load_dataset('json',data_files=dir2)['train']
    # 合并前注意对各属性格式进行统一，尤其Answer
    merged_data = concatenate_datasets([dataset1,dataset2])
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