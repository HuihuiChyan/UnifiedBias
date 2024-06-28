import os
import re
import json
import torch
import argparse
import random
import time
import tqdm
import requests
import pandas as pd
import multiprocessing
import timeout_decorator
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def build_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        choices=("mixtral-8x7b-instruct-v0.1", "llama-2-70b-chat"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        nargs='+',
        choices=("self", "position", "verbosity", "same_verbo"),
        default=None,
    )
    parser.add_argument(
        "--infer-mode",
        type=str,
        choices=("pairwise", "pointwise"),
        default="pairwise",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./test_data",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--rewrite-logit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--process-num",
        type=int,
        default=10,
    )
    return parser

def batched_generation(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("Start load VLLM model!")
    import vllm
    model = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.85)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
        prompt_logprobs=0,
    )
    print("VLLM model loaded!")

    pred_list = model.generate(prompts, sampling_params)

    pred_list = [it.outputs[0].text for it in pred_list]

    return pred_list

def load_dataset(data_type, model_name):
    if data_type == "same_verbo":
        return pd.read_csv(f"test_data/same_verbo.csv")

def calculate_metrics(y_true_list, y_pred_list, infer_type):

    def translate_score_to_win_list(score_list, T=0.0):
        win_list = []
        for i in range(len(score_list)):
            if score_list[i][0] - score_list[i][1] > T:
                win_list.append(1)
            elif score_list[i][1] - score_list[i][0] > T:
                win_list.append(-1)
            else:
                win_list.append(0)
        return win_list

    y_true = translate_score_to_win_list(y_true_list)
    y_pred = translate_score_to_win_list(y_pred_list)

    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

if __name__ == "__main__":

    data_type = "same_verbo"
    model_name = "llama-2-70b-chat"
    infer_mode = "pairwise"


    logit_file = f"output_data/{data_type}-{model_name}-{infer_mode}.jsonl"
    with open(logit_file, "r", encoding="utf-8") as fin:
        lines = [json.loads(line.strip()) for line in fin.readlines()]
        pred_scores = [line["pred_score"] for line in lines]


    relia_file = f"output_data/{data_type}-{model_name}-{infer_mode}-ans-relia.json"
    with open(relia_file, "r", encoding="utf-8") as fin:
        lines = json.load(fin)
        answers = lines["logit"]
    
    import pdb;pdb.set_trace()

    acc = calculate_metrics(answers, pred_scores, infer_mode)
    print(acc)