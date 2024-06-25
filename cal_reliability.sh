export CUDA_VISIBLE_DEVICES=3

python3 -u src/cal_reliability.py \
    --model-name "llama-2-13b-chat" \
    --infer-mode "pairwise" \
    --data-type "pandalm" \
    --max-new-token 512 \
    --output-file "output_data/${MODEL_TYPE}-${DATA_TYPE}-relia.json"

# python3 -u src/evaluate_reliability.py \
#     --model-type ${MODEL_TYPE} \
#     --data-type $DATA_TYPE \
#     --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
#     --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"