python3 -u evaluate_judge.py \
    --model-name "gpt-4-1106-preview" \
    --infer-mode "pointwise" \
    --data-type "position" \
    --process-num 20 \
    --max-new-token 1024

export CUDA_VISIBLE_DEVICES=3
python3 -u evaluate_judge.py \
    --model-name "vicuna-13b" \
    --infer-mode "pairwise" \
    --data-type "self" \
    --max-new-token 1024