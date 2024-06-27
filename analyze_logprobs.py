data_type = "self"
model_name = "llama-2-70b-chat"
infer_mode = "pairwise"
relia_file = f"output_data/{data_type}-{model_name}-{infer_mode}-ans-relia.json"
with open(relia_file, "r") as file_in:
    results = json.loads(file_in)
    import pdb;pdb.set_trace()