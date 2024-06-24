export CUDA_VISIBLE_DEVICES=7
python3 -u evaluate_judge.py \
    --model-name "llama-2-13b-chat" \
    --infer-mode "pairwise" \
    --data-type "verbosity" "position" \
    --max-new-token 1024 \
    --save-logit