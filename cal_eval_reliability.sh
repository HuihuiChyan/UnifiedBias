export CUDA_VISIBLE_DEVICES=3

python3 -u cal_eval_reliability.py \
    --model-name "vicuna-13b" \
    --infer-mode "pairwise" \
    --data-type "pandalm" \
    --max-new-token 512