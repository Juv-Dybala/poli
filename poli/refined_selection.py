import torch
import transformers
from transformers import pipeline,AutoTokenizer,T5ForConditionalGeneration,AutoModelForCausalLM
import os
from tqdm import tqdm

# This file uses T5.


QR2A_PROMPT = "Answer the question: {Question} \n There is a thought you can refer. \n {Rationale} \n " + \
            "So the answer I think is "
Q2A_PROMPT = "Answer the question: {Question} \n The answer I think is "
R2A_PROMPT = "There is a thought of a question. \n {Rationale} \n " + \
            "You can find out the answer of the question in these options: "


def ask_lm(input,model,tokenizer):

    tokenized_input = tokenizer(input,return_tensors="pt",padding=True).to("cuda")
    output_sequences = model.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output
        max_new_tokens = 50
    )

    lm_answer = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]
    return lm_answer


def judge_answer(reply,true_answer):
    # true answer的格式应为 ['(key)','text']
    if len(reply) == 1:
        return true_answer[0][1] == reply
    return true_answer[0] in reply or true_answer[1].lower() in reply.lower()


def select_rationale(model,tokenizer, question, ground_answer, pre_filter_rationales):
    '''
    Template
    Answer the question: What forms beads of water?  (A) Necklaces. (B) Steam. (C) Glass beads . (D) a wave (E) tiny (F) a solute (G) rain (H) Bracelets.
    There is a thought you can refer.
    Explanation: Beads of water are small droplets of water that form on a surface, such as a leaf or a window pane, when the air is cooled.
    This process is called condensation.
    Steam is a form of water vapor that is produced when water is heated, and it can also form beads of water when it comes into contact with a cool surface.
    So the answer I think is
    '''

    # 去重
    pre_filter_rationales = list(set(pre_filter_rationales))

    golden_rationales = []
    pass_or_not = []

    for rationale in pre_filter_rationales:
        if rationale.isspace() or rationale == "":
            continue

        input_dict = {'Question':question,'Rationale':rationale}
        input = QR2A_PROMPT.format_map(input_dict)
        # print(input)

        lm_answer = ask_lm(input,model,tokenizer)
        
        
        # print(lm_answer)
        if judge_answer(lm_answer,ground_answer):
            print(rationale)
            print("{}----Answer correctly!".format(lm_answer))
            golden_rationales.append(rationale)
            pass_or_not.append(True)
        else:
            print(rationale)
            print("{}----Answer wrong.".format(lm_answer))
            pass_or_not.append(False)
        
    print(pass_or_not)

    return golden_rationales


def statistic_las(eval_model,tokenizer,dataset):
    # eval_model:用于评测的模型
    # dataset: Q A R 数据
    result = {}

    # 计算LAS：对每一条QAR都测评 QR->A 和 Q->A ,然后分组统计频数计算ACC 、LAS
    pbar = tqdm(total = len(dataset))
    pbar.set_description("LAS calculating...")
    i = 0
    count_q2a = 0
    count_qr2a = 0

    for item in dataset:
        i += 1

        question = item['Question']
        print(question)
        true_answer = item['Answer']
        print(true_answer)

        q2a_input = Q2A_PROMPT.format_map(item)
        q2a_answer = ask_lm(q2a_input,eval_model,tokenizer)
        print(q2a_answer)
        if judge_answer(q2a_answer,true_answer):
            count_q2a += 1
        
        qr2a_input = QR2A_PROMPT.format_map(item)
        qr2a_answer = ask_lm(qr2a_input,eval_model,tokenizer)
        print(qr2a_answer)
        if judge_answer(qr2a_answer,true_answer):
            count_qr2a += 1
        
        q2a_acc = count_q2a / i
        qr2a_acc = count_qr2a / i
        las = qr2a_acc - q2a_acc

        print(f"Temporarily ACC(Q->A):{q2a_acc}, ACC(QR->A):{qr2a_acc}, LAS:{las}")
        print("=================================")

        pbar.update(1)

    pbar.close()
    result['ACC(Q->A)'] = count_q2a/len(dataset)
    result['ACC(QR->A)'] = count_qr2a/len(dataset)
    result['LAS'] = result['ACC(QR->A)'] - result["ACC(Q->A)"]

    return result


def group_by_leaked(eval_model,tokenizer,dataset):
    # 对rationale进行分组，依据是否可以 R->A 划分为两组

    print("Group by leaked or not. -------------------------------------")

    leakage_rationales = []
    no_leakage_rationales = []

    pbar = tqdm(total = len(dataset))
    pbar.set_description("Grouping...")

    i = 0

    for item in dataset:

        i += 1
        question = item['Question']
        answer = item['Answer']

        # 把选项摘出来
        choices = "(A)" + question.split("(A)")[-1]
        # print(choices)

        input = R2A_PROMPT.format_map(item) + choices + ".\n So the answer I think is "
        print(input)
        print("-------------------")
        reply = ask_lm(input,eval_model,tokenizer)
        print(reply)
        
        leaked = judge_answer(reply,answer)
        if leaked:
            leakage_rationales.append(item)
        else:
            no_leakage_rationales.append(item)

        print("Temporary leaking rate is {}".format(len(leakage_rationales)/i))

        pbar.update(1)

    pbar.close()
    print("Group complete! ")

    return leakage_rationales,no_leakage_rationales
    

def get_rationale_type(reward_model,tokenizer,question,true_answer,rationale):
    
    q2a_input = Q2A_PROMPT.format_map({'Question':question})
    q2a_answer = ask_lm(q2a_input,reward_model,tokenizer)
    print(q2a_answer,end=" ")
    q2a = judge_answer(q2a_answer,true_answer)
        
    qr2a_input = QR2A_PROMPT.format_map({'Question':question,'Rationale':rationale})
    qr2a_answer = ask_lm(qr2a_input,reward_model,tokenizer)
    print(qr2a_answer)
    qr2a = judge_answer(qr2a_answer,true_answer)

    if qr2a:
        rationale_type = 2 if not q2a else 1
    else:
        rationale_type = 0 if not q2a else -1
    
    return rationale_type

