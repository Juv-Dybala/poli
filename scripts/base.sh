python ../poli/data_process.py \
    --large_model "meta-llama/Llama-2-7b-chat-hf" \
    --small_model "google/flan-t5-small" \
    --dataset "qasc" \
    --inference_num 20 \
    --wo_opinion_rate 0.2 \
    --dir_name "base" \
    --step2 \
    # --use_opinion_ft \
    # --sycophancy
    