export CUDA_VISIBLE_DEVICES=4
python3 -u evaluate_judge.py \
    --model-name "llama-2-13b-chat" \
    --infer-mode "pointwise" \
    --data-type "position" \
    --max-new-token 1024