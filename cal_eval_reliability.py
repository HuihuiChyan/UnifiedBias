from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from tqdm import tqdm
import os
import gc
import json
import vllm
import copy
from evaluate_bias import load_dataset, build_dataset, build_params, build_prompt, parse_predictions

def get_multi_answer(
    model_path,
    prompts,
    max_new_token=2048,
    temperature=0.1,
    top_p=1.0,
):
    print("Start load VLLM model!")
    model = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16", gpu_memory_utilization=0.9)
    tokenizer = model.get_tokenizer()
    if "Llama3" in model_path:
        stop_token_ids = [tokenizer.eos_token_id]
    else:
        stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
    )
    print("VLLM model loaded!")

    MAX_LEN = model.llm_engine.model_config.max_model_len - 512
    prompt_ids = [tokenizer.encode(prompt)[-MAX_LEN:] for prompt in prompts]

    pred_list = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)

    prompt_token_ids = [it.prompt_token_ids for it in pred_list]
    output_token_ids = [it.outputs[0].token_ids for it in pred_list]

    prefix_lens = [len(prompt_ids) for prompt_ids in prompt_token_ids]
    target_lens = [len(output_ids) for output_ids in output_token_ids]

    output_tokens = [it.outputs[0].text for it in pred_list]

    output_ids = [ids[0]+ids[1] for ids in zip(prompt_token_ids, output_token_ids)]

    return {"predictions": output_tokens, "prefix_lens": prefix_lens, "target_lens": target_lens, "output_ids": output_ids}

@torch.inference_mode()
def get_single_evaluation(
    model,
    output_ids,
    prefix_len,
    target_len,
):
    # output_ids: The predicted ids consist of both instruction and response, shape is [batch_size, sequence_len]
    # prefix_len: The length of the instruction part
    # target_len: The length of the response part

    # assert output_ids.size()[0] == 1
    output_ids = [torch.as_tensor(oi) for oi in output_ids]
    masked_pos = [(torch.arange(len(output_ids[i])) >= prefix_len[i]).long() for i in range(len(output_ids))]

    pad_token_id = 128001 # tokenizer.convert_tokens_to_ids("<|end_of_text|>")
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

    evaluation_logit = torch.gather(logprobs, dim=-1, index=shifted_output_ids.unsqueeze(-1)).squeeze(-1).sum(-1)
    evaluation_logit = [evaluation_logit[i] / target_len[i] for i in range(len(evaluation_logit))]

    import pdb;pdb.set_trace()

    logprobs_variance = torch.var(logprobs, dim=-1)
    logprobs_variance[output_ids == -100] = 0  # instruction masking
    # averaged on target length
    evaluation_var = logprobs_variance.sum(-1)[0] / target_len

    logprobs[output_ids == -100] = 0  # instruction masking
    # The original entropy has a minus sign, but we remove it to keep the positive correlation
    logprobs_entropy = torch.mean(logprobs * outputs["logits"], dim=-1)
    # averaged on target length
    evaluation_ent = logprobs_entropy.sum(-1)[0] / target_len

    return {"logit": evaluation_logit, "entropy": evaluation_ent, "variance": evaluation_var}

if __name__ == "__main__":
    random.seed(42)
    parser = build_params()
    args = parser.parse_args()

    data_type = args.data_type[0]

    dataset = load_dataset(data_type)
    
    instruction = build_prompt(args.model_name, args.infer_mode)

    prompts, answers = build_dataset(dataset, instruction, args.infer_mode)

    sample_idx = random.randint(0, len(prompts)-1)

    print("********************************Sampled Prompt********************************")
    print(prompts[sample_idx]+"\n")
    print("******************************Sampled Prompt Ended****************************"+"\n")

    eval_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}.jsonl"
    model_path = os.path.join("models", args.model_name)
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as fin:
            lines = [json.loads(line) for line in fin.readlines()]
            predictions = [line["prediction"] for line in lines]
            pred_scores = [line["pred_score"] for line in lines]
            prefix_lens = [line["prefix_len"] for line in lines]
            target_lens = [line["target_len"] for line in lines]
            output_ids = [line["output_ids"] for line in lines]

    else:
        outputs = get_multi_answer(model_path, prompts, args.max_new_token)

        predictions, prefix_lens, target_lens, output_ids = outputs["prediction"],\
                                                            outputs["prefix_lens"],\
                                                            outputs["target_lens"],\
                                                            outputs["output_ids"]

        pred_scores = [parse_predictions(p, args.infer_mode) for p in predictions]
        with open(eval_file, "w", encoding="utf-8") as fout:
            for p in zip(predictions, pred_scores, prefix_lens, target_lens, output_ids):
                pred_line = {"prediction": p[0], "pred_score": p[1], "prefix_len": p[2], "target_len": p[3], "output_ids": p[4]}
                fout.write(json.dumps(pred_line)+"\n")

    print("*******************************Sampled Prediction*****************************")
    print(predictions[sample_idx]+"\n")
    print("****************************Sampled Prediction Ended**************************"+"\n")

    gc.collect()
    torch.cuda.empty_cache()

    # 初始化结果字典
    results = {"Entropy": [], "Variance": [], "Logit": []}

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",).half()
    model.eval()

    for i in tqdm(range(len(predictions)), desc="Calculating reliability score"):
        evaluation = get_single_evaluation(
            model,
            output_ids[i:i+2],
            prefix_lens[i:i+2],
            target_lens[i:i+2],
        )
        logit = evaluation["logit"]
        entropy = evaluation["entropy"]
        variance = evaluation["variance"]
        # 将结果添加到字典中
        results["Logit"].append(logit.item() if isinstance(
            entropy, torch.Tensor) else entropy)
        results["Entropy"].append(entropy.item() if isinstance(
            entropy, torch.Tensor) else entropy)
        results["Variance"].append(variance.item() if isinstance(
            variance, torch.Tensor) else variance)

    # 将所有结果写入 JSON 文件
    relia_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}-eval-relia.json"
    with open(relia_file, "w") as file_out:
        json.dump(results, file_out, indent=4)

    print(f"All reliability scores have been saved to {relia_file}.")