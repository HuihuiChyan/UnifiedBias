export CUDA_VISIBLE_DEVICES=3

python3 -u cal_reliability.py \
    --model-name "llama-2-13b-chat" \
    --infer-mode "pairwise" \
    --data-type "pandalm" \
    --max-new-token 512

# python3 -u evaluate_reliability.py \
#     --model-name "llama-2-13b-chat" \
#     --infer-mode "pairwise" \
#     --data-type "pandalm"