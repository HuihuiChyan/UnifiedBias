export CUDA_VISIBLE_DEVICES=1,2,3,5
python3 -u evaluate_bias.py \
    --model-name "Meta-Llama-3-70B-Instruct" \
    --infer-mode "pairwise" \
    --data-type "pandalm" \
    --max-new-token 1024