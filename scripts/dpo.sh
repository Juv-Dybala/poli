python ../poli/dpo_train.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "qasc" \
    --max_length 1024 \
    --seed 42 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --batch_size 32 \
    --grad_acc_step 4 \
    --warmup 100 \
    --train_epoch 2 \
    --save_steps 400 \
    --lr 3e-4 \
    --sft_dir "" \
    --dir_name "wo+right_fwo_0.2filter" \
    
    # --max_step 20 \
    # --eval_opinion