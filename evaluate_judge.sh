export CUDA_VISIBLE_DEVICES=2,3,4,5

python3 -u evaluate_judge.py \
    --model-name "mixtral-8x7b-instruct-v0.1" \
    --infer-mode "pairwise" \
    --data-type "verbosity" \
    --max-new-token 1024