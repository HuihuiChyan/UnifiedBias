export CUDA_VISIBLE_DEVICES=1

python3 -u cal_ans_reliability.py \
    --model-name "mixtral-8x7b-instruct-v0.1" \
    --infer-mode "pairwise" \
    --data-type "self" \
    --max-new-token 512