import json
import numpy as np
import random
import argparse
from evaluate_bias import calculate_metrics, load_dataset, build_params, parse_predictions


def load_results(file_path):
    """从文件加载分数结果"""
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

def normalize_scores(scores):
    """归一化分数"""
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(score - min_score) /
                         (max_score - min_score) for score in scores]
    return normalized_scores

def compute_combined_score(entropy_scores, variance_scores):
    """计算熵和方差的组合分数"""
    normalized_entropy = normalize_scores(entropy_scores)
    normalized_variance = normalize_scores(variance_scores)
    combined_scores = [(normalized_entropy[i] + normalized_variance[i]
                        ) / 2 for i in range(len(entropy_scores))]
    return combined_scores

def compute_accuracy_rate(metric_results, answers, judge_output, total_length, dataset_type):
    """根据分数结果计算准确率"""
    if dataset_type == "auto-j":
        pass
    else:
        average_scores = metric_results
        sorted_indices = np.argsort(np.array(average_scores))
        top_half_indices = sorted_indices[:len(sorted_indices) // 2]
        accuracy_rate = calculate_metrics(
            [answers[i] for i in top_half_indices], [judge_output[i] for i in top_half_indices], dataset_type)
    return accuracy_rate

def compute_bucketing_rate(metric_results, answers, judge_output, total_length, dataset_type):
    """根据分数结果计算准确率"""
    if dataset_type == "auto-j":
        pass
    else:
        bucket_num = 5
        bucket_rate = {}
        average_scores = metric_results
        sorted_indices = np.argsort(np.array(average_scores))
        bucket_size = len(sorted_indices)//bucket_num
        for i in range(bucket_num):
            top_half_indices = sorted_indices[bucket_size*i:bucket_size*(i+1)]
            accuracy_rate = calculate_metrics(
                [answers[i] for i in top_half_indices], [judge_output[i] for i in top_half_indices], dataset_type)
            bucket_rate[str(i)] = accuracy_rate
            print(f"Bucket {i}: {accuracy_rate}")
    return bucket_rate


def main():
    random.seed(42)
    np.random.seed(42)

    parser = build_params()
    args = parser.parse_args()

    data_type = args.data_type[0]

    dataset = load_dataset(data_type)
    answers = [example["score"] for example in dataset]

    relia_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}-relia.json"
    relia_scores = load_results(relia_file)["Logit"]
    # relia_scores = compute_combined_score(relia_scores["Entropy"], relia_scores["Variance"])

    logit_file = f"output_data/{data_type}-{args.model_name}-{args.infer_mode}.jsonl"

    with open(logit_file, "r", encoding="utf-8") as fin:
        lines = [json.loads(line.strip()) for line in fin.readlines()]
        predictions = [line["prediction"] for line in lines]

    pred_scores = [parse_predictions(p, args.infer_mode) for p in predictions]

    accuracy_rate = compute_accuracy_rate(relia_scores, answers, pred_scores, len(relia_scores), data_type)

    print(f"Accuracy Rate: {accuracy_rate}")

    bucket_rate = compute_bucketing_rate(relia_scores, answers, pred_scores, len(relia_scores), data_type)

    # 随机选取等量的索引作为一个随机基线比较
    random_indices = np.random.choice(
        len(answers), len(relia_scores)//2, replace=False)

    random_accuracy_rate = calculate_metrics(
        [answers[i] for i in random_indices], [pred_scores[i] for i in random_indices], data_type)
    print(f"Random Selection Accuracy Rate: {random_accuracy_rate}")


if __name__ == "__main__":
    main()