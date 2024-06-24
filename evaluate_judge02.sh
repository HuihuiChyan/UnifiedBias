python3 -u evaluate_judge.py \
    --model-name "gpt-3.5-turbo-0613" \
    --infer-mode "pairwise" \
    --data-type "position" \
    --process-num 10 \
    --max-new-token 1024 \
    --save-logit