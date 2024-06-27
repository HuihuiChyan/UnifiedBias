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

    token_ids_a = tokenizer(prompts_a)["input_ids"]
    token_ids_b = tokenizer(prompts_b)["input_ids"]
    token_ids_prefix = tokenizer(prompts_prefix)["input_ids"]

    output_ids = token_ids_a + token_ids_b
    prefix_lens = [len(t) for t in token_ids_prefix]
    whole_lens_a = [len(t) for t in token_ids_a]
    whole_lens_b = [len(t) for t in token_ids_b]
    target_lens_a = [t[0]-t[1] for t in zip(whole_lens_a, prefix_lens)]
    target_lens_b = [t[0]-t[1] for t in zip(whole_lens_b, prefix_lens)]
    
    prefix_lens = prefix_lens * 2
    target_lens = target_lens_a + target_lens_b

    return output_ids, prefix_lens, target_lens

if __name__ == "__main__":
    random.seed(42)
    parser = build_params()
    args = parser.parse_args()

    data_type = args.data_type[0]

    dataset = load_dataset(data_type, args.model_name)

    # 初始化结果字典
    results = {"logit": [], "entropy": [], "variance": []}

    model_path = os.path.join("models", args.model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    output_ids, prefix_lens, target_lens = build_dataset(dataset["los"], tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").half()
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