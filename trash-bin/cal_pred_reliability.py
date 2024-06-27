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
    stop_token_ids = [tokenizer.eos_token_id]
    # if "Llama3" in model_path:
    #     stop_token_ids = [tokenizer.eos_token_id]
    # else:
    #     stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
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
    tokenizer,
    max_length,
    output_ids,
    prefix_len,
    target_len,
):
    # output_ids: The predicted ids consist of both instruction and response, shape is [batch_size, sequence_len]
    # prefix_len: The length of the instruction part
    # target_len: The length of the response part

    output_ids = [torch.as_tensor(oi) for oi in output_ids]
    masked_pos = [(torch.arange(len(output_ids[i])) >= prefix_len[i]).long() for i in range(len(output_ids))]

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    # pad to max_length first to avoid OOM during inference
    output_ids[0] = torch.nn.ConstantPad1d((0, max_length - output_ids[0].shape[0]), pad_token_id)(output_ids[0])
    masked_pos[0] = torch.nn.ConstantPad1d((0, max_length - masked_pos[0].shape[0]), 0)(masked_pos[0])
    output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
    masked_pos = torch.nn.utils.rnn.pad_sequence(masked_pos, batch_first=True, padding_value=0)

    output_ids = output_ids.to(model.device)
    masked_pos = masked_pos.to(model.device)

    outputs = model(
        input_ids=output_ids.to(model.device),
        output_hidden_states=True,
        output_attentions=True,
    )

    # the predict ids should be shifted left
    shifted_output_ids = torch.roll(output_ids, shifts=-1)
    logprobs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)

    logprobs = logprobs * masked_pos.unsqueeze(-1)

    evaluation_logit = torch.gather(logprobs, dim=-1, index=shifted_output_ids.unsqueeze(-1)).squeeze(-1).sum(-1)
    evaluation_logit = [evaluation_logit[i] / target_len[i] for i in range(len(evaluation_logit))]

    logprobs_variance = torch.var(logprobs, dim=-1)
    # averaged on target length
    evaluation_var = [logprobs_variance.sum(-1)[i] / target_len[i] for i in range(len(logprobs_variance))]

    # The original entropy has a minus sign, but we remove it to keep the positive correlation
    logprobs_entropy = torch.mean(logprobs * outputs["logits"], dim=-1)
    # averaged on target length
    evaluation_ent = [logprobs_entropy.sum(-1)[i] / target_len[i] for i in range(len(logprobs_variance))]

    return {"logit": evaluation_logit, "entropy": evaluation_ent, "variance": evaluation_var}

if __name__ == "__main__":
    random.seed(42)
    parser = build_params()
    args = parser.parse_args()

    data_type = args.data_type[0]

    dataset = load_dataset(data_type)

    instruction = instruction = "[INST]\n{prompt} [/INST]"

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
    results = {"logit": [], "entropy": [], "variance": []}

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").half()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    batch_size = 4
    max_length = max([l[0]+l[1] for l in zip(prefix_lens, target_lens)])
    for i in tqdm(range(0, len(predictions), batch_size), desc="Calculating reliability score"):
        evaluation = get_single_evaluation(
            model,
            tokenizer,
            max_length,
            output_ids[i:i+batch_size],
            prefix_lens[i:i+batch_size],
            target_lens[i:i+batch_size],
        )
        # 将结果添加到字典中
        results["logit"].extend(evaluation["logit"])
        results["entropy"].extend(evaluation["entropy"])
        results["variance"].extend(evaluation["variance"])

    # 将所有结果写入 JSON 文件
    relia_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}-pred-relia.json"
    with open(relia_file, "w") as file_out:
        json.dump(results, file_out, indent=4)

    print(f"All reliability scores have been saved to {relia_file}.")