import torch
import transformers
from transformers import pipeline,AutoTokenizer,T5ForConditionalGeneration,AutoModelForCausalLM
import os
from tqdm import tqdm
import re

# This file uses T5.


QR2A_PROMPT = "Answer the question: {Question} \n There is a thought you can refer. \n {Rationale} \n " + \
            "So the answer I think is "
Q2A_PROMPT = "Answer the question: {Question} \n The answer I think is "
R2A_PROMPT = "There is a thought of a question. \n {Rationale} \n " + \
            "You can find out the answer of the question in these options: "

EXAMPLE_PROMPT = "Question: What form on angiosperms? (A) lamphreys (B) backbones (C) flowers (D) pigment (E) coliform (F) adult (G) antibodies (H) Testes" + \
                    "Answer: (C) \n" + \
                "Question: What parents abandon their eggs? (A) lamphreys (B) platypus (C) deer (D) Unsafe (E) mammal (F) vorticella (G) reptile (H) jellyfish" + \
                    "Answer: (G) \n" + \
                "Question: Where are nutrients held? (A) reefs (B) marine (C) saturated (D) tissue (E) forests (F) Earth (G) aquatic (H) flagella" + \
                    "Answer: (D) \n" # + \
                # "Question: what is less lightweight than cartilage but stronger? (A) skin (B) cilia (C) tissue (D) weater (E) adult (F) Type O (G) Mohs (H) bone" + \
                #     "Answer: (H) \n" + \
                # "Question: What is more pliable than bone? (A) Cartilage (B) tiny hairs (C) tetraceratops (D) teeth (E) femur (F) mineral (G) Therapsids (H) keratin" + \
                #     "Answer: (A) \n"



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


def ask_lm_prob(input,model,tokenizer,true_answer):
    # print(input)
    tokenized_input = tokenizer(input,return_tensors="pt",padding=True).to("cuda")
    output = model.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output
        max_new_tokens = 20,
        return_dict_in_generate=True, # generate logits
        output_scores = True,
    )

    # output.scores是一个tuple，元素数=生成的字符数（开始符不算），每一个元素都是这一次字符在词表（32128个词）上的得分
    # 这个得分是对数似然，softmax转为概率
    score = torch.stack(output.scores,dim=0).squeeze() # [seq_len-1,vocab_size]
    score = score.softmax(-1)
    # print(score)

    # 与词表配合查看，看看生成概率高的是哪些词
    # tops,ids = torch.topk(score,k=10,dim=1)
    # print(tops)

    output_sequence = output.sequences
    lm_answer = tokenizer.batch_decode(output_sequence, skip_special_tokens=True)[0]
    # print(lm_answer,end=" ")
    # print(output_sequence)

    answerKey = true_answer[0][1]
    
    if re.search(r"\([A-Z].*\)",lm_answer): #（A）格式
        loc = torch.nonzero(torch.eq(output_sequence[0],get_vocab_loc(tokenizer,"▁(")))[0]
        answerKey_index = get_vocab_loc(tokenizer,answerKey)
        prob = score[loc,answerKey_index]
        # total = 0.0
        # for answer in ['A','B','C','D','E','F','G','H']:  # 记录各选项概率
        #     answer_index = get_vocab_loc(tokenizer,answer)
        #     answer_prob = score[loc,answer_index].item()
        #     print(f"({answer})--{answer_prob}")
        #     total += answer_prob
        # print("Total prob -----------------{}".format(total))
    elif re.search(r"Option\s[A-Z]",lm_answer): # Option A格式
        loc = torch.nonzero(torch.eq(output_sequence[0],get_vocab_loc(tokenizer,"▁Option")))[0]
        answerKey_index = get_vocab_loc(tokenizer,"▁"+ answerKey)
        prob = score[loc,answerKey_index]
    elif true_answer[1].lower() == lm_answer.lower(): # Text大小写格式，直接找概率最高者
        prob,_ = torch.topk(score[0],k=1)
    else:
        answerKey_index = get_vocab_loc(tokenizer,"▁"+ answerKey)
        prob = score[0,answerKey_index]
    prob = prob.item()
    # print (prob)
    return lm_answer,prob


def get_vocab_loc(tokenizer,target_word):
    # 得到target_word在词表中的索引

    vocab = tokenizer.get_vocab()
    # print(target_word)
    # print(vocab)
    # 通过观察词表，发现在T5中，独立的词(如开头词)是需要加下划线 ▁ 的(这个和打出来的下划线_不一样...)
    # sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    # print(sorted_vocab)
    return vocab[target_word]


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


def q2a(model,tokenizer,question,true_answer,prob=False,few_shot=True):

    example = "Answer the following multiple choice question and follow this format: \n" + EXAMPLE_PROMPT

    if few_shot:
        q2a_input = example + "Question: " + question + " \n Answer: "
    else:
        q2a_input = "Question: " + question + " \n Answer: "
    if not prob:
        q2a_answer = ask_lm(q2a_input,model,tokenizer)
        print(q2a_answer,end=" ")
        q2a = judge_answer(q2a_answer,true_answer)
        return q2a
    else:
        q2a,prob = ask_lm_prob(q2a_input,model,tokenizer,true_answer)
        return prob


def qr2a(model,tokenizer,question,true_answer,rationale,prob=False,few_shot=True):

    example = "Answer the following multiple choice question and follow this format: \n" + EXAMPLE_PROMPT
    
    if few_shot:
        qr2a_input = example + "Question: " + question + "\n Explanation: " + rationale + "\n Answer: "
    else:
        qr2a_input = "Question: " + question + "\n Explanation: " + rationale + "\n Answer: "
    if not prob:
        qr2a_answer = ask_lm(qr2a_input,model,tokenizer)
        print(qr2a_answer)
        qr2a = judge_answer(qr2a_answer,true_answer)
        return qr2a
    else:
        qr2a,prob = ask_lm_prob(qr2a_input,model,tokenizer,true_answer)
        return prob