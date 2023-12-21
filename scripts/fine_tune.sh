python ../poli/fine_tune.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "gsm8k" \
    --math \
    --max_length 512 \
    --seed 42 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --batch_size 32 \
    --grad_acc_step 4 \
    --warmup 100 \
    --train_epoch 5 \
    --save_strategy 'epoch' \
    --save_steps 400 \
    --lr 3e-4 \
    --pretrain_dir "" \
    --dir_name "step1_wo10+right10" \
    
    # --max_step 20 \
    # --eval_opinion