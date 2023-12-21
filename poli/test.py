from data_process import *
from datasets_load import *
from data_utils import *
from datasets import load_dataset,DatasetDict

dataset_name = "gsm8k"
large_model_name = "meta-llama/Llama-2-7b-chat-hf"
small_model_name = "google/flan-t5-large"


# filter_threshold(dataset_name,"prob_right10_large",threshold=0.1)
# filter_threshold(dataset_name,"prob_wo10_large",threshold=0.1)
# join_processed_dataset(dataset_name,dir1="prob_wo10_large_0.1filter",
#                         dir2="prob_right10_large_0.1filter",
#                         joined_dir="prob_wo10+right10_large_0.1filter")
# filter_threshold_failed(dataset_name,"prob_wo10_large_failed",threshold=-0.1)
generate_dpo_data(dataset_name,chosen_dir="prob_wo10+right10_large_0.05filter",
                  rejected_dir="prob_wo10_large_failed_-0.05filter",
                  out_dir="wo+right_fwo_0.05filter")
exit()



# generate_ft_data(dataset_name,"prob_wo10+right10_large_0.2filter")

# generate_ft_data(dataset_name,"prob_right10_large_0.2filter")
# generate_ft_data(dataset_name,"prob_wo10_large_0.2filter")
# merge_dataset(dir_list=["../data/finetuning/qasc/prob_right10_large_0.2filter.jsonl",
#                         "../data/finetuning/qasc/prob_wo10_large_0.2filter.jsonl"],
#                 merged_dir="../data/finetuning/qasc/large0.2.jsonl")
# group_ft_data(dataset_name,"large0.01")
# select_best_worst(dataset_name,"prob_right10_large")

dataset1 = load_finetuning_data(dataset_name,"large0.01")
dataset2 = load_finetuning_data(dataset_name,"prob_right10_large_best")
dataset2 = dataset2.filter(lambda x:x["Rationale"] not in dataset1["Rationale"])
print(len(dataset2))
merge_dataset = concatenate_datasets([dataset1,dataset2])
merge_dataset.to_json("../data/finetuning/qasc/large0.01+best.jsonl")



step2_selection_prob_failed(dataset_name,"step1_wo10_failed",small_model_name,"prob_wo10_large_failed",threshold=1)


# step2_selection_prob(dataset_name,"step1_wo10",small_model_name,"prob_wo10_base",threshold=-1)
# step2_selection_prob(dataset_name,"step1_right10",small_model_name,"prob_right10_base",threshold=-1)
# step2_selection_prob(dataset_name,"step1_wrong_0.5sample",small_model_name,"prob-base_wrong7",threshold=-1)

filter_threshold(dataset_name,"prob_right10_large",threshold=0.01)
filter_threshold(dataset_name,"prob_wo10_large",threshold=0.01)


select_best_worst(dataset_name,dir_name="prob_right10_large")
merge_dataset(["../data/finetuning/qasc/large0.01.jsonl",
               "../data/finetuning/qasc/prob_right10_large_best.jsonl"],
               merged_dir="../data/finetuning/qasc/large0.01+best.jsonl")

step1_generate(large_model_name,dataset_name,inference_num={'wrong':21})
step1_generate(large_model_name,dataset_name,inference_num={'wo':10})


generate_ft_data(dataset_name,"prob-0.25_wrong7")



statistic_failed_generation(large_model_name,small_model_name,dataset_name)




merge_dataset(['../data/finetuning/qasc/prob-0.25_wo6.jsonl',
               '../data/finetuning/qasc/prob-0.25_right7.jsonl',
               '../data/finetuning/qasc/prob-0.25_wrong7.jsonl'
               ],
               '../data/finetuning/qasc/prob-0.25.jsonl')





question = "what can erosion cause? (A) hurricanes (B) h2o (C) Pollution (D) slow and inefficient (E) unstable terrain (F) Physical weathering (G) flooding (H) hydrogen"
answer = ["(E)", "unstable terrain"]
rationale = " Erosion can cause unstable terrain by removing the natural support systems of the land, such as vegetation and soil, which can lead to landslides, rockfalls, and other types of instability.\n This can have serious consequences, including loss of life and property damage.\n\nExplanation:\nErosion is the process of wearing away the Earth's surface by natural forces such as wind, water, or ice.\n It can cause a range of problems, including:\nUnstable terrain: Erosion can remove the natural support systems of the land, such as vegetation and soil, which can lead to landslides, rockfalls, and other types of instability.\n This can have serious consequences, including loss of life and property damage.\n\Hurricanes: Erosion can also contribute to the formation of hurricanes by weakening the structural integrity of the land.\n  When the land is weakened, it can be more susceptible to the strong winds and heavy rainfall associated with hurricanes, which can lead to further erosion and damage.\n\n   H2o: Erosion can also cause changes in the water table, which can lead to flooding and other hydrological problems.\n\nPollution: Erosion can also lead to the transport of pollutants, such as pesticides and fertilizers, which can contaminate waterways and harm aquatic ecosystems.\n\nSlow and inefficient: Erosion can also slow down and make inefficient the processes of nutrient cycling, water filtration, and other ecological processes that are essential for maintaining healthy ecosystems.\n\nPhysical weathering: Erosion can also cause physical weathering, which is the breakdown of rocks and soil into smaller particles.\n This can lead to changes in the landscape and can affect the stability of structures"
model,tokenizer = load_t5(small_model_name)

t = get_rationale_type(model,tokenizer,question,answer,rationale)
print(t)

q2a_prob = q2a(model,tokenizer,question,answer,prob=True)
qr2a_prob = qr2a(model,tokenizer,question,answer,rationale,prob=True)

print(q2a_prob)
print(qr2a_prob)


step1_generate(large_model_name,dataset_name,inference_num={'wrong':14})
# filter_threshold(dataset_name,"prob_base_wo6",-0.5)
# filter_threshold(dataset_name,"prob_base_right7",-0.5)
# filter_threshold(dataset_name,"prob_base_wrong7",-0.5)
# select_best_worst(dataset_name,"prob_base_wo6")
exit()











step1_generate(large_model_name,dataset_name,inference_num={"wo":10,"right":10})

sizes = ["small"] # ,"base","large"]
for size in sizes:
    print(f"{size} =================================================================")
    small_model_name = f"google/flan-t5-{size}"
    step2_selection(dataset_name,"step1_wo_0.6sample",small_model_name,f"type_wo6_{size}")
    step2_selection(dataset_name,"step1_right_0.7sample",small_model_name,f"type_right7_{size}")
    step2_selection(dataset_name,"step1_wrong_0.5sample",small_model_name,f"type_wrong7_{size}")
   



count_reward = {-1:0, 0:0, 1:0, 2:0}
dataset = load_ppo_data(dataset_name,"step1_wo6")

for item in dataset:
    count_reward[item['Reward']] += 1

print(count_reward)

# inference_num = {'wo':6,'right':7,'wrong':7}
# step1_generate(large_model_name,dataset_name,inference_num)
wo_dir = "../data/finetuning/qasc/step1_wo_0.6sample.jsonl"
right_dir = "../data/finetuning/qasc/step1_right_0.7sample.jsonl"
wrong_dir = "../data/finetuning/qasc/step1_wrong_0.5sample.jsonl"

# generate_ft_data(dataset_name,dir_name="step1_wrong_0.5sample")

merge_dataset(dir1=wo_dir,dir2=right_dir,merged_dir="../data/finetuning/qasc/wo6+right7.jsonl")

merge_dataset(dir1="../data/finetuning/qasc/wo6+right7.jsonl",dir2=wrong_dir,
              merged_dir="../data/finetuning/qasc/union677.jsonl")


# dataset_name = "ai2_arc"
# datasets_load(dataset_name,'ARC-Easy')
# generate_ft_data("qasc","step1_wrong")

# sample_rationales(dataset_name,"step1_wo",0.6)
# sample_rationales(dataset_name,"step1_right",0.7)
# sample_rationales(dataset_name,"step1_wrong",0.5)
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
















