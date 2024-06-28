export CUDA_VISIBLE_DEVICES=2,3,4,5

python3 -u cal_ans_reliability.py \
    --model-name "llama-2-70b-chat" \
    --infer-mode "pairwise" \
    --data-type "same_verbo" \
    --max-new-token 512