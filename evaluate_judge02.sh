export CUDA_VISIBLE_DEVICES=3
python3 -u evaluate_judge.py \
    --model-name "vicuna-13b" \
    --infer-mode "pairwise" \
    --data-type "self" \
    --max-new-token 1024