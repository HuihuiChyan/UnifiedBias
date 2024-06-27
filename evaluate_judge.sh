export CUDA_VISIBLE_DEVICES=1,6,7

python3 -u evaluate_judge.py \
    --model-name "llama-2-70b-chat" \
    --infer-mode "pairwise" \
    --data-type "self" \
    --max-new-token 1024