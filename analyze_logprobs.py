import json
from collections import defaultdict

data_type = "self"
model_name = "llama-2-70b-chat"
infer_mode = "pairwise"
relia_file = f"output_data/{data_type}-{model_name}-{infer_mode}-ans-relia.json"

heat_dict = defaultdict(lambda x:[])

with open(relia_file, "r") as file_in:
    results = json.loads(file_in)
    logprobs = results["prompt_logprobs"]
    heat_dict = {}
    for logprob in logprobs:
        for i, lp in enumerate(logprob):
            heat_dict[str(i)].append(lp)
    
    heat_list = []
    for i in range(10000):
        if heat_dict[i] != []:
            heat_list = sum(heat_dict[str(i)]) / len(heat_dict[str(i)])

    print(heat_list)