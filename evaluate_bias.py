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
        choices=("gpt-4-1106-preview", "gpt-3.5-turbo-0613", "llama-2-13b-chat", "vicuna-13b", "Meta-Llama-3-70B"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,\
        nargs='+',
        choices=("self", "position", "verbosity", "pandalm", "judgelm"),
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


@torch.inference_mode()
def batched_generation(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("Start load VLLM model!")
    import vllm
    model = vllm.LLM(model=model_path, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.8)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
    )
    print("VLLM model loaded!")

    pred_list = model.generate(prompts, sampling_params)
    pred_list = [it.outputs[0].text for it in pred_list]

    return pred_list

@timeout_decorator.timeout(60)
def do_one_request(url, headers, data):
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    res = response['choices'][0]['message']['content'].strip()

    return res

def request_gpt(prompt, model, temperature, max_new_tokens):
    # url = "https://www.qwopenai.com/v1/chat/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "Bearer sk-15g5tdDeJ25iG5wX1e0790C8Bf69458dB827D9D04c66Db78",
    # }
    url = "https://api.ai-gaochao.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0",
    }
    max_tries = 5
    res = ''
    response = None
    # sys_info = {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."}    
    for i in range(max_tries):
        try:
            data = {
                "model": model, 
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            res = do_one_request(url, headers, data)
            break
        except Exception as e:
            print("Exception! The response is " + str(response))
            time.sleep(5)
            continue
    return res

def gpt_scoring(prompt, model, temperature, max_new_tokens):

    prediction = request_gpt(prompt, model, temperature=temperature, max_new_tokens=max_new_tokens)

    counter.value += 1
    print(f"gpt_scoring {counter.value} finished.")

    return prediction

def load_dataset(data_type, data_path = "./test_data"):

    if data_type == "judgelm":
        with open(os.path.join(data_path, "judgelm/judgelm_val_5k.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]

        with open(os.path.join(data_path, "judgelm/judgelm_val_5k_gpt4.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset_score = [json.loads(line) for line in lines]

        new_dataset = []
        for example, example_score in zip(dataset, dataset_score):
            example["prompt"] = example["question_body"]
            example["response_a"] = example["answer1_body"]
            example["response_b"] = example["answer2_body"]
            example["score"] = example_score["score"]

            if example["score"] != [-1, -1]:
                new_dataset.append(example)
        
        dataset = new_dataset
        dataset = random.sample(dataset, k=1000)

    elif data_type == "pandalm":
        with open(os.path.join(data_path, "pandalm/testset-v1.json"), "r") as fin:
            lines = json.load(fin)

        dataset = []
        for index, line in enumerate(lines):

            # if index >= 100:
            #     break

            example = {}
            if line["input"].strip() == "":
                example["prompt"] = line["instruction"]
            else:
                example["prompt"] = line["input"] + \
                    "\n" + line["instruction"]
            example["response_a"] = line["response1"]
            example["response_b"] = line["response2"]
            if line["annotator1"] == line["annotator2"] or line["annotator1"] == line["annotator2"]:
                example["score"] = line["annotator1"]
            elif line["annotator2"] == line["annotator3"]:
                example["score"] = line["annotator2"]
            else:
                example["score"] = random.choice(
                    [line["annotator1"], line["annotator2"], line["annotator3"]])
            # unify the score to judgelm format
            score_mapping = {"0": [1, 1], "1": [1, 0], "2": [0, 1]}
            example["score"] = score_mapping[str(example["score"])]
            dataset.append(example)

    return dataset

def build_prompt(model_name, infer_mode):
    if "gpt" in model_name:
        if infer_mode == "pairwise":
            prompt = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]"""

        elif infer_mode == "pointwise":
            prompt = """[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
[Question]
{question}
[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]"""

    elif model_name == "vicuna-13b":
        if infer_mode == "pairwise":
            prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
ASSISTANT:"""
        elif infer_mode == "pointwise":
            prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
[Question]
{question}
[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]
ASSISTANT:"""

    else:
        if infer_mode == "pairwise":
            prompt = """[INST]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer] [/INST]"""

        elif infer_mode == "pointwise":
            prompt = """[INST]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
[Question]
{question}
[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer] [/INST]"""        

    return prompt

def parse_predictions(review, infer_mode):

    if infer_mode == "pairwise":
        if "[[A]]" in review or "[A]" in review:
            return [1, 0]
        elif "[[B]]" in review or "[B]" in review:
            return [0, 1]
        elif "[[C]]" in review  or "[C]" in review:
            return [1, 1]
        else:
            return [0, 0]

    elif infer_mode == "pointwise":
        if "Rating: [[" in review:
            pos = review.rfind("Rating: [[")
            pos2 = review.find("]]", pos)
            assert pos != -1 and pos2 != -1
            return float(review[pos + len("Rating: [["):pos2].strip())
        elif "Rating: [" in review:
            pos = review.rfind("Rating: [")
            pos2 = review.find("]", pos)
            assert pos != -1 and pos2 != -1
            return float(review[pos + len("Rating: ["):pos2].strip())
        elif "Rating: " in review:
            pos = review.rfind("Rating: ")
            score = re.search(r"[0-9\.]+", review[pos + len("Rating: ")])
            if score is not None:
                return float(score.group())
            else:
                return 5.1
        else:
            return 5.1

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
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # add metrics to dict
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    return metrics_dict

def build_dataset(dataset, instruction, infer_mode):

    prompts = []
    answers = []
    for index, example in enumerate(dataset):

        if infer_mode == "pairwise":
            prompt = instruction.format(question=example["prompt"],
                                        answer_a=example["response_a"],
                                        answer_b=example["response_b"])
            prompts.append(prompt)

        elif infer_mode == "pointwise":
            prompt_a = instruction.format(question=example["prompt"],
                                          answer=example["response_a"])
            prompt_b = instruction.format(question=example["prompt"],
                                          answer=example["response_b"])
            prompts.append(prompt_a)
            prompts.append(prompt_b)

        answers.append(example["score"])
    
    return prompts, answers
    

def init(c):
    global counter
    counter = c

if __name__ == "__main__":

    random.seed(42)
    parser = build_params()
    args = parser.parse_args()
    result_dicts = {}

    data_type = args.data_type[0]

    dataset = load_dataset(data_type)
    
    instruction = build_prompt(args.model_name, args.infer_mode)

    prompts, answers = build_dataset(dataset, instruction, args.infer_mode)

    print("********************************Sampled Prompt********************************")
    print(prompts[random.randint(0, len(prompts)-1)]+"\n")
    print("******************************Sampled Prompt Ended****************************"+"\n")

    logit_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}.jsonl"
    if not args.rewrite_logit and os.path.exists(logit_file):
        with open(logit_file, "r", encoding="utf-8") as fin:
            lines = [json.loads(line.strip()) for line in fin.readlines()]
            predictions = [line["prediction"] for line in lines]
    else:
        if "gpt" not in args.model_name:
            predictions = batched_generation(os.path.join("models", args.model_name), 
                                             prompts,
                                             max_new_token=args.max_new_token,
                                             temperature=args.temperature,
                                             top_p=args.top_p)
        else:
            manager = multiprocessing.Manager()
            counter = manager.Value("counter", 0)
            pool = multiprocessing.Pool(processes=args.process_num, initializer=init, initargs=(counter,))
            
            len_prompts = len(prompts)
            print(f"Totally {len_prompts} prompts.")

            if args.process_num == 1:
                predictions = [gpt_scoring(sample, model=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_token)
                                for sample in prompts]
            else:
                pool_fn = partial(gpt_scoring, model=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_token)
                predictions = pool.map(pool_fn, prompts)
                pool.close()

    pred_scores = [parse_predictions(p, args.infer_mode) for p in predictions]
    with open(f"output_data/{data_type}-{args.model_name}-{args.infer_mode}.jsonl", "w", encoding="utf-8") as fout:
        for p in zip(predictions, pred_scores):
            pred_line = {"prediction": p[0], "pred_score": p[1]}
            fout.write(json.dumps(pred_line)+"\n")

    if args.infer_mode == "pointwise":
        predictions_a = [pred for pred in predictions[0::2]]
        predictions_b = [pred for pred in predictions[1::2]]
        pred_scores_a = [pred for pred in pred_scores[0::2]]
        pred_scores_b = [pred for pred in pred_scores[1::2]]
        predictions = [[pred[0], pred[1]] for pred in zip(predictions_a, predictions_b)]
        pred_scores = [[pred[0], pred[1]] for pred in zip(pred_scores_a, pred_scores_b)]

    metrics_dict = calculate_metrics(answers, pred_scores, args.infer_mode)

    print(metrics_dict)