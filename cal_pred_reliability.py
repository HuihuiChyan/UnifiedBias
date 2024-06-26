from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from tqdm import tqdm
import os
import gc
import json
import vllm
import copy
from evaluate_bias import load_dataset, build_dataset, build_params

def get_multi_answer(
    model_path,
    prompts,
    max_new_token=2048,
    temperature=0.1,
    top_p=1.0,
):
    print("Start load VLLM model!")
    if "Llama3" in model_path:
        stop_token_ids = [tokenizer.eos_token_id]
    else:
        stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    model = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16", gpu_memory_utilization=0.8)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
    )
    print("VLLM model loaded!")

    tokenizer = model.get_tokenizer()
    MAX_LEN = model.llm_engine.model_config.max_model_len - 512
    prompt_ids = [tokenizer.encode(prompt)[-MAX_LEN:] for prompt in prompts]

    pred_list = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)

    prompt_token_ids = [it.prompt_token_ids for it in pred_list]
    output_token_ids = [it.outputs[0].token_ids for it in pred_list]

    prefix_lens = [len(prompt_ids) for prompt_ids in prompt_token_ids]
    target_lens = [len(output_ids) for output_ids in output_token_ids]

    output_tokens = [it.outputs[0].text for it in pred_list]

    output_ids = [ids[0]+ids[1] for ids in zip(prompt_token_ids, output_token_ids)]

    return output_tokens, prefix_lens, target_lens, output_ids

@torch.inference_mode()
def get_single_evaluation(
    model,
    output_ids_ori,
    prefix_len,
    target_len,
):
    # output_ids_ori: The predicted ids consist of both instruction and response, shape is [1, sequence_len]
    # prefix_len: The length of the instruction part
    # target_len: The length of the response part

    assert output_ids_ori.size()[0] == 1
    output_ids_ori = output_ids_ori.to(model.device)

    input_ids = copy.deepcopy(output_ids_ori)
    output_ids = output_ids_ori.clone()
    output_ids[0][:prefix_len] = -100  # instruction masking
    outputs = model(
        input_ids=torch.as_tensor(input_ids),
        labels=output_ids,
        output_hidden_states=True,
        output_attentions=True,
    )
    # the predict ids should be shifted left
    shifted_input_ids = torch.roll(input_ids, shifts=-1)
    logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)

    logprobs_variance = torch.var(logprobs, dim=-1)
    logprobs_variance[output_ids == -100] = 0  # instruction masking
    # averaged on target length
    evaluation_var = logprobs_variance.sum(-1)[0] / target_len

    logprobs[output_ids == -100] = 0  # instruction masking
    # The original entropy has a minus sign, but we remove it to keep the positive correlation
    logprobs_entropy = torch.mean(logprobs * outputs["logits"], dim=-1)
    # averaged on target length
    evaluation_ent = logprobs_entropy.sum(-1)[0] / target_len

    evaluation_logit = torch.gather(logprobs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    evaluation_logit = evaluation_logit.sum(-1)[0] / target_len

    return {"logit": evaluation_logit, "entropy": evaluation_ent, "variance": evaluation_var}

def build_prompt(model_name, infer_mode):
    if "gpt" in model_name:
        instruction = "{prompt}"

    elif model_name == "vicuna-13b":
        instruction = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {prompt}
ASSISTANT:"""

    elif model_name == "llama2-13b-chat":
        instruction = "[INST]\n{prompt} [/INST]"

    elif model_name == "Meta-Llama-3-70B":    
        instruction = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return instruction


if __name__ == "__main__":
    random.seed(42)
    parser = build_params()
    args = parser.parse_args()

    data_type = args.data_type[0]

    dataset = load_dataset(data_type)

    instruction = build_prompt(args.model_name, args.infer_mode)

    prompts = []
    for example in dataset:
        prompts.append(instruction.format(prompt=example["prompt"]))

    sample_idx = random.randint(0, len(prompts)-1)

    print("********************************Sampled Prompt********************************")
    print(prompts[sample_idx]+"\n")
    print("******************************Sampled Prompt Ended****************************"+"\n")

    model_path = os.path.join("models", args.model_name)
    predictions, prefix_lens, target_lens, output_ids = get_multi_answer(model_path, prompts, args.max_new_token)

    print("*******************************Sampled Prediction*****************************")
    print(predictions[sample_idx]+"\n")
    print("****************************Sampled Prediction Ended**************************"+"\n")

    gc.collect()
    torch.cuda.empty_cache()

    # 初始化结果字典
    results = {"Entropy": [], "Variance": [], "Logit": []}

    model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
    model.eval()

    for i in tqdm(range(len(predictions)), desc="Calculating reliability score"):
        evaluation = get_single_evaluation(
            model,
            torch.as_tensor([output_ids[i]]),
            prefix_lens[i],
            target_lens[i],
        )
        logit = evaluation["logit"]
        entropy = evaluation["entropy"]
        variance = evaluation["variance"]
        # 将结果添加到字典中
        results["Logit"].append(entropy.item() if isinstance(
            entropy, torch.Tensor) else entropy)
        results["Entropy"].append(entropy.item() if isinstance(
            entropy, torch.Tensor) else entropy)
        results["Variance"].append(variance.item() if isinstance(
            variance, torch.Tensor) else variance)

    # 将所有结果写入 JSON 文件
    relia_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}-pred-relia.json"
    with open(relia_file, "w") as file_out:
        json.dump(results, file_out, indent=4)

    print(f"All reliability scores have been saved to {relia_file}.")