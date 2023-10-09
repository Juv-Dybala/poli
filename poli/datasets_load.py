import os
from datasets import load_dataset


def dataset_download(dataset_name):
    # 下载数据集到本地
    dataset = load_dataset(dataset_name)
    print(dataset)
    for split,ds in dataset.items():
        dataset_save_directory = os.path.join("../data","raw",dataset_name,f"{split}.json")
        ds.to_json(dataset_save_directory)
    print("Download {} dataset successfully! ------------------------")


def datasets_load(dataset_name,split='Train'):
    print(f"Loading {dataset_name}[{split}] dataset. ----------------------------")
    # 需要的col：
    # formatted_question、choices（text+label）、answerKey

    if dataset_name == "qasc":
        return load_qasc(dataset_name,split)
    # TODO：其他数据集之后往这里加


def load_qasc(dataset_name,split='Train'):
    
    dataset_dir = os.path.join("../data/raw",dataset_name,"{}.json".format(split))
    dataset = load_dataset("json",data_files=dataset_dir)['train']
    # features: ['id', 'question', 'choices', 'answerKey',
    #           'fact1', 'fact2', 'combinedfact', 'formatted_question']
    dataset = dataset.select_columns(['formatted_question','choices','answerKey'])
    return dataset


def load_preprocessed_data(dir_name):
    # cols:'Question', 'Num of choice', 'Answer', 'Rationales'
    dataset_dir = os.path.join("../data/processed",f"{dir_name}.jsonl")
    dataset = load_dataset('json',data_files=dataset_dir)['train']
    print(dataset)
    return dataset

    