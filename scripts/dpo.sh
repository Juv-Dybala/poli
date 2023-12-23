python ../poli/dpo_train.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "qasc" \
    --max_length 1024 \
    --seed 42 \
    --beta 0.1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --batch_size 16 \
    --grad_acc_step 8 \
    --warmup 100 \
    --train_epoch 1 \
    --save_steps 400 \
    --lr 3e-4 \
    --sft_dir "large-0.4" \
    --dir_name "wo+right_fwo_0.1filter" \
    --eval_ckpts \
    
    # --max_step 20 \
    # --eval_opinion