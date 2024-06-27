export CUDA_VISIBLE_DEVICES=1,2,3,4
python3 -u evaluate_judge.py \
    --model-name "Mixtral-8x7B-Instruct-v0.1" \
    --infer-mode "pairwise" \
    --data-type "self" \
    --max-new-token 1024