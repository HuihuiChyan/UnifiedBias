export CUDA_VISIBLE_DEVICES=2

python3 -u cal_ans_reliability.py \
    --model-name "Mixtral-8x7B-Instruct-v0.1" \
    --infer-mode "pairwise" \
    --data-type "verbosity" \
    --max-new-token 512