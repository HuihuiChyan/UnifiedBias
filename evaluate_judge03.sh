export CUDA_VISIBLE_DEVICES=7
python3 -u evaluate_judge.py \
    --model-name "vicuna-13b" \
    --infer-mode "pairwise" \
    --data-type "self" \
    --max-new-token 1024 \
    --save-logit