from data_process import model_download,step2_selection,generate_ft_data,statistic_leakage_data,STaR
from data_process import *
from datasets_load import *
from datasets import load_dataset,DatasetDict

dataset_name = "qasc"
large_model_name = "meta-llama/Llama-2-7b-chat-hf"
inference_num = {'wo':6,'right':7,'wrong':7}
step1_generate(large_model_name,dataset_name,inference_num)

exit()

# dataset_name = "ai2_arc"
# datasets_load(dataset_name,'ARC-Easy')
# generate_ft_data("qasc","step1_wrong")
wo_dir = "../data/processed/qasc/step1_wo.jsonl"
right_dir = "../data/processed/qasc/step1_right.jsonl"
wrong_dir = "../data/processed/qasc/step1_wrong.jsonl"

wo_right = "../data/processed/qasc/intsec_wo&right.jsonl"
intersect_dataset(dir1=right_dir,dir2=wrong_dir,filter_attribute="Question")



merge_dataset(dir1="../data/processed/STaR_0.5sample.jsonl",
            dir2="../data/other/step1_notIn_star.jsonl",
            merged_dir="../data/processed/star+step1.jsonl")


subtract_dataset(dir1="../data/processed/step1.jsonl",
                 dir2="../data/processed/STaR.jsonl",
                 filter_attribute="Question",
                 subtracted_dir="../data/other/step1_notIn_star.jsonl")
sample_rationales("STaR")


# format_answer("step1")
# format_answer("self_consistency1")
STaR(large_model_name,dataset_name)
duplicate_hard_question_rationales(dataset_name,dir_name="../data/finetuning/hard_more.jsonl")

dir_name = "self_consistency1"
# generate_ft_data("qasc",dir_name)
# step2_selection("step1","google/flan-t5-large","just_test")
statistic_leakage_data("google/flan-t5-large",dir_name)
print("==============================================================")
statistic_leakage_data("google/flan-t5-large",dir_name,grouping=False)
print("==============================================================")
statistic_leakage_data("google/flan-t5-small",dir_name)

# 比较两组数据差异
large = load_dataset("json",data_files="../data/finetuning/T5filter_large.jsonl")['train']
base = load_dataset("json",data_files="../data/finetuning/T5filter_base.jsonl")['train']
print(large)
print(base)
print("large_only ==========================================================")
only_large = large.filter(lambda x:x['Rationale'] not in base['Rationale'])
only_large.to_json("./large_only.json")
for item in only_large:
    print(item)
    print("-------------------------")

print("base_only ==========================================================")
only_base = base.filter(lambda x:x['Rationale'] not in large['Rationale'])
only_base.to_json("./base_only.json")
for item in only_base:
    print(item)
    print("-------------------------")


# 下载模型并filter step2

sizes = ["small","base","large"]
for size in sizes:
    print(f"{size} =================================================================")
    small_model_name = f"google/flan-t5-{size}"
    output_dir = f"T5filter_{size}"
    # model_download(small_model_name)
    step2_selection("self_consistency1",small_model_name,output_dir)
    generate_ft_data("qasc",output_dir)
















