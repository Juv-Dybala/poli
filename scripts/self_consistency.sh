python ../poli/data_process.py \
    --large_model "meta-llama/Llama-2-7b-chat-hf" \
    --small_model "google/flan-t5-small" \
    --dataset "qasc" \
    --inference_num 20 \
    --wo_opinion_rate 1.0 \
    --dir_name "self_consistency" \
    --step2 \
    # --qa2r 1\
    # --use_opinion_ft \
    # --sycophancy
    