export CUDA_VISIBLE_DEVICES=1

python3 -u cal_eval_reliability.py \
    --model-name "Meta-Llama-3-70B-Instruct" \
    --infer-mode "pairwise" \
    --data-type "pandalm" \
    --max-new-token 512