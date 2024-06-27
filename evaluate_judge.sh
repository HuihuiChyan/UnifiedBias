export CUDA_VISIBLE_DEVICES=2,3,4,5

python3 -u evaluate_judge.py \
    --model-name "llama-2-70b-chat" \
    --infer-mode "pairwise" \
    --data-type "self" \
    --max-new-token 1024