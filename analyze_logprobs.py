import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

data_type = "self"
model_name = "llama-2-70b-chat"
infer_mode = "pairwise"
relia_file = f"output_data/{data_type}-{model_name}-{infer_mode}-ans-relia.json"

heat_dict = defaultdict(lambda :[])

with open(relia_file, "r") as file_in:
    results = json.load(file_in)
    logprobs = results["prompt_logprobs"]
    for logprob in logprobs:
        for i, lp in enumerate(logprob):
            heat_dict[str(i)].append(lp)

    heat_list = []
    for i in range(10000):
        for j in range(10):
            if heat_dict[str(i*10+j)] != []:
                heat_list.append(sum(heat_dict[str(i)]) / len(heat_dict[str(i)]))

import pdb;pdb.set_trace()
heat_list = np.array(heat_list)
posi_list = np.arange(len(heat_list))
plt.plot(posi_list, heat_list)  # Plot the chart
plt.show()  # display