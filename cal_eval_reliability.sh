export CUDA_VISIBLE_DEVICES=2

python3 -u cal_eval_reliability.py \
    --model-name "llama-2-13b-chat" \
    --infer-mode "pairwise" \
    --data-type "pandalm" \
    --max-new-token 512