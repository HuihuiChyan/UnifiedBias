export CUDA_VISIBLE_DEVICES=4

python3 -u cal_eval_reliability.py \
    --model-name "llama-2-13b-chat" \
    --infer-mode "pairwise" \
    --data-type "judgelm" \
    --max-new-token 512

python3 -u eval_reliability.py \
    --model-name "llama-2-13b-chat" \
    --infer-mode "pairwise" \
    --data-type "judgelm"