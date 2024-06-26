export CUDA_VISIBLE_DEVICES=1,2,3,5
python3 -u evaluate_bias.py \
    --model-name "llama-2-13b-chat" \
    --infer-mode "pairwise" \
    --data-type "pandalm" \
    --max-new-token 1024