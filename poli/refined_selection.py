import torch
import transformers
from transformers import pipeline,AutoTokenizer,T5ForConditionalGeneration,AutoModelForCausalLM
import os

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

        input = "Answer the question: " + question + "\n There is a thought you can refer." + rationale + \
                "So the answer I think is"
        # print(input)

        lm_answer = ask_lm(input,model,tokenizer)
        
        
        # print(lm_answer)
        if ground_answer[0] in lm_answer or ground_answer[1].lower() in lm_answer.lower():
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


