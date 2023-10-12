from data_process import model_download,step2_selection,generate_ft_data,statistic_leakage_data
from datasets import load_dataset,DatasetDict

dir_name = "self_consistency1"
# generate_ft_data("qasc",dir_name)
# step2_selection("step1","google/flan-t5-large","just_test")
statistic_leakage_data("google/flan-t5-large",dir_name)
print("==============================================================")
statistic_leakage_data("google/flan-t5-large",dir_name,grouping=False)
print("==============================================================")
statistic_leakage_data("google/flan-t5-small",dir_name)


exit()


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
# large_model_name = "meta-llama/Llama-2-7b-chat-hf"
sizes = ["small","base","large"]
for size in sizes:
    print(f"{size} =================================================================")
    small_model_name = f"google/flan-t5-{size}"
    output_dir = f"T5filter_{size}"
    # model_download(small_model_name)
    step2_selection("self_consistency1",small_model_name,output_dir)
    generate_ft_data("qasc",output_dir)
















