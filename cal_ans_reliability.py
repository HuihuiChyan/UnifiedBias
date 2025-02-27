from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from tqdm import tqdm
import os
import gc
import json
import vllm
import copy
from evaluate_judge import load_dataset, build_params, calculate_metrics, calculate_bias_diff

def batched_evaluation(
    model_path,
    prompts,
    temperature=0.0,
    top_p=1.0,
):
    print("Start load VLLM model!")
    model = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.8)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=1,
        top_p=top_p,
        prompt_logprobs=0,
    )
    print("VLLM model loaded!")

    pred_list = model.generate(prompts, sampling_params)

    
    # prompt_logprobs = [[list(lp.items())[0][1] for lp in pl.prompt_logprobs[1:]] for pl in pred_list]
    prompt_logprobs = []
    for pl in pred_list:
        if pl == None:
            prompt_logprobs.append([0])
        else:
            prompt_logprobs.append([list(lp.items())[0][1] for lp in pl.prompt_logprobs[1:]])

    return prompt_logprobs

def build_eval_dataset(dataset, tokenizer):

    instruction_prefix = "[INST]\n{prompt}[/INST]"
    instruction = "[INST]\n{prompt}[/INST]{answer}"

    prompts = []
    answers = []
    prefix_lens = []
    for index, example in dataset.iterrows():

        prompts.append(instruction.format(prompt=example["prompt"], answer=example["response_a"]))
        prompts.append(instruction.format(prompt=example["prompt"], answer=example["response_b"]))

        prompts_prefix = instruction_prefix.format(prompt=example["prompt"])
        token_ids_prefix = tokenizer(prompts_prefix)["input_ids"]
        prefix_lens.append(len(prompts_prefix))
        prefix_lens.append(len(prompts_prefix))

        if example["winner_model_a"] == 1:
            answers.append([1, 0])
        elif example["winner_model_b"] == 1:
            answers.append([0, 1])
        else:
            answers.append([1, 1])

    sample_idx = random.randint(0, len(prompts)-2)

    print("********************************Sampled Prompt********************************")
    print(prompts[sample_idx]+"\n")
    print(prompts[sample_idx+1]+"\n")
    print("******************************Sampled Prompt Ended****************************"+"\n")

    return prompts, prefix_lens, answers

if __name__ == "__main__":
    random.seed(42)
    parser = build_params()
    args = parser.parse_args()

    data_type = args.data_type[0]

    dataset = load_dataset(data_type, args.model_name)

    # 初始化结果字典
    results = {}

    model_path = os.path.join("models", args.model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # prompts_win, prefix_lens_win, answers_win = build_eval_dataset(dataset["win"], tokenizer)
    # prompts_los, prefix_lens_los, answers_los = build_eval_dataset(dataset["los"], tokenizer)

    # prompts = prompts_win + prompts_los
    # prefix_lens = prefix_lens_win + prefix_lens_los
    # answers = answers_win + answers_los

    prompts, prefix_lens, answers = build_eval_dataset(dataset, tokenizer)

    prompt_logprobs = batched_evaluation(
        model_path,
        prompts,
        temperature=0.0,
        top_p=1.0,
    )

    pred_scores = [sum(pl)/len(pl) for pl in prompt_logprobs]

    pred_scores_a = [pred for pred in pred_scores[0::2]]
    pred_scores_b = [pred for pred in pred_scores[1::2]]
    pred_scores = [[pred[0], pred[1]] for pred in zip(pred_scores_a, pred_scores_b)]

    # win_acc = calculate_metrics(answers[:len(answers)//2], pred_scores[:len(answers)//2], args.infer_mode)
    # los_acc = calculate_metrics(answers[len(answers)//2:], pred_scores[len(answers)//2:], args.infer_mode)
    # bias_diff = calculate_bias_diff(answers[:len(answers)//2], pred_scores[:len(answers)//2], answers[len(answers)//2:], pred_scores[len(answers)//2:])
    # result_dicts = {"win_acc": win_acc, "los_acc": los_acc, "diff": win_acc-los_acc, "bias_diff": bias_diff}

    print(calculate_metrics(answers, pred_scores, args.infer_mode))

    # 将所有结果写入 JSON 文件
    relia_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}-ans-relia.json"
    results = {"logit": pred_scores, "prompt_logprobs": prompt_logprobs}
    with open(relia_file, "w") as file_out:
        json.dump(results, file_out, indent=4)