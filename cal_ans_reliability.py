from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from tqdm import tqdm
import os
import gc
import json
import vllm
import copy
from evaluate_judge import load_dataset, build_params

def batched_evaluation(
    model_path,
    prompts,
    max_new_token=16,
    temperature=0.0,
    top_p=1.0,
):
    print("Start load VLLM model!")
    model = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=1,
        top_p=top_p,
        prompt_logprobs=0,
    )
    print("VLLM model loaded!")

    pred_list = model.generate(prompts, sampling_params)

    prompt_logprobs = [[list(lp.items())[0][1] for lp in pl.prompt_logprobs[1:]] for pl in pred_list]

    prompt_logprobs = [sum(pl) for pl in prompt_logprobs]

    return prompt_logprobs

def build_dataset(dataset, tokenizer):

    instruction_prefix = "[INST]\n{prompt}[/INST]"
    instruction = "[INST]\n{prompt}[/INST]{answer}"

    prompts_prefix = []
    prompts_a = []
    prompts_b = []
    for index, example in dataset.iterrows():
        prompts_prefix.append(instruction_prefix.format(prompt=example["prompt"]))
        prompts_a.append(instruction.format(prompt=example["prompt"], answer=example["response_a"]))
        prompts_b.append(instruction.format(prompt=example["prompt"], answer=example["response_b"]))

    sample_idx = random.randint(0, len(prompts_prefix)-1)

    print("********************************Sampled Prompt********************************")
    print(prompts_prefix[sample_idx]+"\n")
    print(prompts_a[sample_idx]+"\n")
    print(prompts_b[sample_idx]+"\n")
    print("******************************Sampled Prompt Ended****************************"+"\n")

    token_ids_prefix = tokenizer(prompts_prefix)["input_ids"]

    prefix_lens = [len(t) for t in token_ids_prefix]

    prefix_lens = prefix_lens * 2
    prompts = prompts_a + prompts_b

    return prompts, prefix_lens

if __name__ == "__main__":
    random.seed(42)
    parser = build_params()
    args = parser.parse_args()

    data_type = args.data_type[0]

    dataset = load_dataset(data_type, args.model_name)

    # 初始化结果字典
    results = {"logit": []}

    model_path = os.path.join("models", args.model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts, prefix_lens = build_dataset(dataset["los"], tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").half()
    model.eval()

    batch_size = 8
    model_path = os.path.join("models", "mixtral-8x7b-instruct-v0.1")
    results = batched_evaluation(
        model_path,
        prompts,
        max_new_token=16,
        temperature=0.0,
        top_p=1.0,
    )

    # 将所有结果写入 JSON 文件
    relia_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}-pred-relia.json"
    with open(relia_file, "w") as file_out:
        json.dump(results, file_out, indent=4)

    print(f"All reliability scores have been saved to {relia_file}.")