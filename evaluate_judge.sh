export CUDA_VISIBLE_DEVICES=1,2,3,4,5,7

python3 -u evaluate_judge.py \
    --model-name "llama-2-70b-chat" \
    --infer-mode "pairwise" \
    --data-type "verbosity" \
    --max-new-token 1024