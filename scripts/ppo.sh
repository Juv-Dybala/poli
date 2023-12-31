python ../poli/ppo_train.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "qasc" \
    --reward_model "google/flan-t5-small" \
    --max_length 512 \
    --seed 42 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --batch_size 16 \
    --lr 1.41e-5 \
    --dir_name "rerun677" \