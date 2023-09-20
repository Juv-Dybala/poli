python ../SILM/fine_tune.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "qasc" \
    --max_length 1024 \
    --seed 42 \
    --lora_r 16 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --batch_size 64 \
    --grad_acc_step 4 \
    --warmup 5 \
    --max_step 1000 \
    --lr 2e-4 \
    # --dir_name "self_consistency1" \ 
    # --eval_opinion