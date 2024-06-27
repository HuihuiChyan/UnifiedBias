export CUDA_VISIBLE_DEVICES=3

python3 -u eval_reliability.py \
    --model-name "Meta-Llama-3-70B-Instruct" \
    --infer-mode "pairwise" \
    --data-type "pandalm" \
    --relia-type "eval"