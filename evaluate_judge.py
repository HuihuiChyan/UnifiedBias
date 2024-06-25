import os
import json
import torch
import argparse
import random
import time
import requests
import pandas as pd
import multiprocessing
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def build_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        choices=("gpt-4-1106-preview", "gpt-3.5-turbo-0613", "llama-2-13b-chat", "vicuna-13b"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,\
        nargs='+',
        choices=("self", "position", "verbosity"),
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

def request_gpt(prompt, model, temperature, max_new_tokens):
    url = "https://www.qwopenai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-15g5tdDeJ25iG5wX1e0790C8Bf69458dB827D9D04c66Db78",
    }
    # url = "https://api.ai-gaochao.cn/v1/chat/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "Bearer sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0",
    # }
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
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response = response.json()
            res = response['choices'][0]['message']['content'].strip()
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

def load_dataset(data_type, model_name):
    if data_type == "self":
        win_data = pd.read_csv(f"test_data/self_win_{model_name}.csv")
        los_data = pd.read_csv(f"test_data/self_los_{model_name}.csv")
    
    elif data_type == "verbosity":
        win_data = pd.read_csv("test_data/verbo_win.csv")
        los_data = pd.read_csv("test_data/verbo_los.csv")

    elif data_type == "position":
        win_data = pd.read_csv("test_data/left_win.csv")
        los_data = pd.read_csv("test_data/left_los.csv")
    
    data = {"win": win_data, "los": los_data}
    return data

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
        if "[[" in review:
            pos = review.rfind("[[")
            pos2 = review.find("]]", pos)
            assert pos != -1 and pos2 != -1
            return float(review[pos + len("[["):pos2].strip())
        else:
            return 5.0

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

    if infer_type == "pairwise":
        y_true = translate_score_to_win_list(y_true_list)
        y_pred = translate_score_to_win_list(y_pred_list)
    else:
        y_true = y_true_list
        y_pred = y_pred_list

    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

def calculate_bias_diff(y_true_list1, y_pred_list1, y_true_list2, y_pred_list2):

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

    y_true1 = translate_score_to_win_list(y_true_list1)
    y_pred1 = translate_score_to_win_list(y_pred_list1)

    y_true2 = translate_score_to_win_list(y_true_list2)
    y_pred2 = translate_score_to_win_list(y_pred_list2)

    y_pos = sum([int(y[0] == y[1]) for y in zip(y_true1, y_pred1)]) + sum([int(y[0] == -y[1]) for y in zip(y_true2, y_pred2)])
    y_neg = sum([int(y[0] == -y[1]) for y in zip(y_true1, y_pred1)]) + sum([int(y[0] == y[1]) for y in zip(y_true2, y_pred2)])

    bias_diff = (y_pos - y_neg)/(len(y_true1) + len(y_true2))

    return bias_diff

def build_dataset(dataset, instruction, infer_mode):

    for index, example in dataset.iterrows():
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

        assert example["winner_model_a"] + example["winner_model_b"] + example["winner_tie"] == 1

        if example["winner_model_a"] == 1:
            answers.append([1, 0])
        elif example["winner_model_b"] == 1:
            answers.append([0, 1])
        else:
            answers.append([1, 1])
    
    return prompts, answers
    

def init(c):
    global counter
    counter = c

if __name__ == "__main__":

    random.seed(42)
    parser = build_params()
    args = parser.parse_args()
    result_dicts = {}
    for data_type in args.data_type:
        data = load_dataset(data_type, args.model_name)
        
        instruction = build_prompt(args.model_name, args.infer_mode)

        prompts = []
        answers = []
        for data_split in ["win", "los"]:
            
            dataset = data[data_split]

            prompts, answers = build_dataset(dataset, instruction, args.infer_mode)

            print("********************************Sampled Prompt********************************")
            print(prompts[random.randint(0, len(prompts)-1)]+"\n")
            print("******************************Sampled Prompt Ended****************************"+"\n")

        logit_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}.jsonl"
        if not args.rewrite_logit and os.path.exists(logit_file):
            with open(logit_file, "r", encoding="utf-8") as fin:
                lines = [json.loads(line.strip()) for line in fin.readlines()]
                predictions = [line["prediction"] for line in lines]
                pred_scores = [line["pred_score"] for line in lines]
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

        win_acc = calculate_metrics(answers[:len(answers)//2], pred_scores[:len(answers)//2], args.infer_mode)
        los_acc = calculate_metrics(answers[len(answers)//2:], pred_scores[len(answers)//2:], args.infer_mode)
        bias_diff = calculate_bias_diff(answers[:len(answers)//2], pred_scores[:len(answers)//2], answers[len(answers)//2:], pred_scores[len(answers)//2:])
        result_dicts[data_type] = {"win_acc": win_acc, "los_acc": los_acc, "diff": win_acc-los_acc, "bias_diff": bias_diff}

        print(result_dicts)

    for data_type in args.data_type:
        print("*****************Results**********************")
        print(f"Model: {args.model_name}, Data: {data_type}, Infer: {args.infer_mode}")
        print(result_dicts[data_type])
        print("**********************************************")