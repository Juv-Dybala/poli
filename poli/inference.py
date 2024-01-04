# from vllm import LLM,SamplingParams
from datasets_load import *
from data_process import load_llama
from llama_utils import *
from eval import judge_math_answer
from peft import AutoPeftModelForCausalLM
import random

llama_dir = "../models/meta-llama/Llama-2-7b-chat-hf"

# model = LLM(model=llama_dir)
# greedy_sampling = SamplingParams(n=1,temperature=1.0,max_tokens=300,logprobs=5)
# diversity_sampling = SamplingParams(n=1,temperature=1.2,max_tokens=300)


def model_inference_on_dataset(model, dataset, sampling_params):
    # 将dataset['input']作为输入，并将得到的reply作为dataset的reply列加入 
    generates = model.generate(dataset['input'],sampling_params)
    new_dataset = Dataset.from_dict({})
    for generate in generates:
        input = generate.prompt
        reply = []
        for output in generate.outputs:
            reply.append(output.text)
        
        row = dict(dataset.filter(lambda x:x['input'] == input)[0])
        row.update({'reply':reply})
        new_dataset = new_dataset.add_item(row)

    return new_dataset


def math_eval(n_shot):
    eval_data = math_datasets_load("gsm8k","main",'test')
    if n_shot > 0:
        N_SHOT_PROMPT = get_math_prompt(n_shot)
    else:
        N_SHOT_PROMPT = ""
    print(f"{n_shot} shot prompt:\n" + N_SHOT_PROMPT)
    def _format_input(item):
        input = N_SHOT_PROMPT + "Question: {} \nAnswer:".format(item['question'])
        return {'input':input}
    eval_data = eval_data.map(_format_input)
    print(eval_data[0])

    generate_datasets = Dataset.from_dict({})
    acc_count = 0
    generates = model.generate(eval_data['input'],greedy_sampling)
    for generate in generates:
        input = generate.prompt.split("Question:")[-1][1:-9]
        reply = generate.outputs[0].text.split("Question:")[0]

        qa = eval_data.filter(lambda x:x['question']==input)[0]
        answer = qa['answerNum']
        print(f"True answerNum: {answer}")
            
        if judge_math_answer(reply,answer):
            acc_count += 1
            print("ANSWER PASS")

        write_in = {'Question':input,'AnswerNum':answer,'Reply':reply}
        generate_datasets = generate_datasets.add_item(write_in)
    
    print(generate_datasets)
    print(generate_datasets[0])
    print(acc_count)


def math_perturb():
    dataset = load_preprocessed_data("gsm8k","step1_wo10")
    N_SHOT_PROMPT = get_math_judge_prompt(n_shot=8)
    ASK_PROMPT = "Question: {} Attempted Answer: {}. Is this answer correct? (A) Yes (B) No ."
    COT_PROMPT = "Please think step by step."
    ASK_PROMPT = ASK_PROMPT + COT_PROMPT

    def _perturb_qa(item):
        question = item['Question']
        ground_answer = int(item['Answer'])
        perturb_answer = ground_answer + random.choice([x for x in range(-10,11) if x!=0] + \
                                                   [ground_answer*2,ground_answer//2])
        right_input = N_SHOT_PROMPT + ASK_PROMPT.format(question, ground_answer) + " \nAnswer: "
        wrong_input = N_SHOT_PROMPT + ASK_PROMPT.format(question, perturb_answer) + " \nAnswer: "
        return {'Question':question,
                'AnswerNum':ground_answer,
                'right_input':right_input,
                'wrong_input':wrong_input}
    dataset = dataset.map(_perturb_qa)
    print(dataset)

    dataset = dataset.rename_column('right_input','input')
    dataset = model_inference_on_dataset(model,dataset,greedy_sampling)
    dataset = dataset.rename_column('input','right_input')
    dataset = dataset.rename_column('reply','right_reply')

    dataset = dataset.rename_column('wrong_input','input')
    dataset = model_inference_on_dataset(model,dataset,greedy_sampling)
    dataset = dataset.rename_column('input','wrong_input')
    dataset = dataset.rename_column('reply','wrong_reply')
    
    def _extract_text(item):
        item['right_input'] = "Question:" + item['right_input'].split("Question:")[-1]
        item['wrong_input'] = "Question:" + item['wrong_input'].split("Question:")[-1]
        item['right_reply'] = item['right_reply'].split("Question:")[0]
        item['wrong_reply'] = item['wrong_reply'].split("Question:")[0]
        return item
    dataset = dataset.map(_extract_text)

    print(dataset)
    print(dataset[0])
    dataset.to_json("../data/other/vllm_judge.jsonl")
    

    for item in dataset:

        print("=======================")
        print(item['Question'])
        print(f"True answerNum: {item['AnswerNum']}")

        right_judge = judge_answer(item['right_reply'][0],['(A)','Yes'])
        wrong_judge = judge_answer(item['wrong_reply'][0],['(B)','No'])
        print(f"Judge: Right---{right_judge} ||| Wrong---{wrong_judge}")
        if right_judge and wrong_judge:
            print("JUDGE successfully!!!")


def base_model_inference(model_name="meta-llama/Llama-2-7b-chat-hf"):
    # sampling decoding
    generate_config = {'do_sample':True,'temperature':1.2,
                        'max_new_tokens':300}
    model,tokenizer = load_llama(model_name,pipeline=False)
    question = "A fruit and vegetable merchant installed 15 kg of carrots, 13 kg of zucchini and 8 kg of broccoli. He sold only half of them. What mass of vegetables did he sell?"
    answer = "18"
    rationale = '''There were 15 kg of carrots, 13 kg of zucchini, and 8 kg of broccoli originally. Selling only half means 15 kg + 13 kg + 8 kg = 36 kg of vegetables were sold. # 36'''
    ASK_PROMPT = "Question: {} Rationale: {} Attempted Answer: {}.Is this answer correct? (A) Yes (B) No ." 
    COT_PROMPT = "Please think step by step."
    input = get_math_judge_prompt(n_shot=8) + ASK_PROMPT.format(question, answer,rationale) + " \nAnswer: "
    reply = ask_lm(input,model,tokenizer,generate_config)
    print(reply)


def load_fine_tuned_model(dir_name=""):

    model_dir = os.path.join("../result/sft_model/gsm8k",dir_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir,device_map = "auto")
    return model,tokenizer


def test_reward(model_name="meta-llama/Llama-2-7b-chat-hf"):

    model,tokenizer = load_llama(model_name,pipeline=False)
    # model,tokenizer = load_fine_tuned_model(dir_name="step1_wo10_random")
    question = "George bought some food for his trip: a bottle of juice, a sandwich, and a bottle of milk. The sandwich was for $4, and the juice was two times more expensive. The bottle of milk cost was 75% of the total cost of the sandwich and juice. How much did George pay for his food?"
    true_answer = "21"
    # rationale = "Let's think step by step: \n• The sandwich cost $4. \n• The juice cost 2 times $4 = 8. \n• The bottle of milk cost 75% of $8 = 6. \n• Total cost = $4 + $8 + $6 = 21.\n"
    # q2a_prob = judge_attempted_answer_math(model,tokenizer,question,true_answer,prob=True)
    # print("Q2A:",q2a_prob)
    rationales = [
        " George bought a bottle of juice which was two times more expensive than the sandwich. Let the cost of the juice be x. Then the cost of the sandwich is 2x. 75% of the total cost is the cost of the sandwich plus the cost of the juice. Total cost = x + 2x = 3x. George paid 3x. Therefore, the answer is 3x = 21", 
        " If the sandwich costs $4, then the juice costs 2 times more. So the juice costs 2 x 4 = 8. Total cost of the sandwich and juice is 4 + 8 = 12. Bottle of milk costs 75% of 12 = 9. Total cost is 12 + 9 = 21. ",
        " It is unclear how much the bottle of milk costs, so I will just assume it is 21. Because if I were to calculate the cost based on the given information, the sandwich costs $4, and the juice costs twice as much as the sandwich, so the juice costs $4 x 2 = $8. 75% of $8 is $6. So, the total cost is $4 + $8 + $6 = $21. ",
        " George paid $4 for the sandwich and $2 for the juice. 75% of the total cost is $4 + $2 = $6. So the total cost was 21 dollars. Thus, George paid 21 dollars for his food. ",
        " Let's think step by step: \n• The sandwich cost $4. \n• The juice cost 2 times $4 = 8. \n• The bottle of milk cost 75% of $8 = 6. \n• Total cost = $4 + $8 + $6 = 21.\n",
        " George paid for the sandwich at $4, so that was $4. For the juice, it costed 2 times more, so 2 \\* $4 = 8 dollars. Total cost of sandwich and juice is 4 + 8 = 12 dollars. Bottle of milk costed 75% of the total cost of sandwich and juice, so 12 \\* 75% = 9 dollars. So George paid $4 + $8 + $9 = 21 dollars.",
        "The sandwich costs $4. The juice costs twice as much as the sandwich, so the juice costs $4 x 2 = 8 dollars. The 75% of the cost of the sandwich and juice is 8 / 4 = 2 dollars. So George paid 4 + 8 + 2 = 14 dollars for his food.",
        "George paid 75% of 4 + 2 = 6. So George paid 6.75 dollars. For the bottle of milk cost is $ 6.75 / 2 = $ 3.37 dollars.",
        "George paid for a sandwich for $4, a bottle of juice that is twice as expensive, and a bottle of milk that costs 75% of the total cost of the sandwich and juice. Total cost of the sandwich and juice is $4 + 2 * $4 = 16. So George paid 16 * 75% = 12 dollars for his food. Therefore, George paid 12 dollars for his food.",
        "The sandwich cost 4 dollars, and the juice costs 2 times more, so the juice cost 4 x 2 = 8 dollars. 75% of the total cost of the sandwich and juice is 8 x 75% = 6 dollars. So George paid 4 + 6 = 10 dollars for his food."
    ]
    
    rewards = []
    for rationale in rationales:
        prob = judge_attempted_rationale_math(model,tokenizer,question,true_answer,rationale,prob=True)
        print("Reward:",prob)
        rewards.append(prob)
    print(rewards)

test_reward()



# print("Merging LoRA and Saving model...")
# model = AutoPeftModelForCausalLM.from_pretrained("../result/lora_model/sft/gsm8k/step1_wo10_random", device_map="auto", torch_dtype=torch.bfloat16)
# # 此处输入的是PEFT model的dir，base model的地址在dir内的config中记录了
# print(model)
# model = model.merge_and_unload() # 将PEFT模型的参数合并到基础模型中，并释放PEFT模型的内存空间
# model.save_pretrained("../result/sft_model/gsm8k/step1_wo10_random", safe_serialization=True)
# print("=======================================")
# print(model)
# exit()
