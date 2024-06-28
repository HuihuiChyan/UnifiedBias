import ast
import copy
import random
import pandas as pd
import numpy as np

df = pd.read_csv("orig_data/train.csv")

# 去除掉轮数超过1的数据
df["round"] = df.apply(lambda x:len(ast.literal_eval(x.prompt)), axis=1)
df = df[df["round"]==1]

def convert_response(x):
    try:
        x = ast.literal_eval(x)[0]
    except:
        x = "null"
    return x

# 将list数据转化为str
df["prompt"] = df["prompt"].apply(lambda x:ast.literal_eval(x)[0])
df["response_a"] = df["response_a"].apply(convert_response)
df["response_b"] = df["response_b"].apply(convert_response)

# 去除掉response为null的数据
df = df[(df["response_a"] != "null") & (df["response_b"] != "null")]

# 对不符合utf-8的数据进行转化
df["prompt"] = df["prompt"].apply(lambda x:str(x).encode('utf-8', 'replace').decode('utf-8'))
df["response_a"] = df["response_a"].apply(lambda x:str(x).encode('utf-8', 'replace').decode('utf-8'))
df["response_b"] = df["response_b"].apply(lambda x:str(x).encode('utf-8', 'replace').decode('utf-8'))

# all_models = sorted(list(set(df["model_a"].to_numpy().tolist())))
# print(all_models) # 共计 64个模型

# all_models_count = {}
# for model_name in all_models:
#     all_models_count[model_name] = len(df[(df["model_a"] == model_name) | (df["model_b"] == model_name)])

# print(all_models_count)

# 我们选择了比较有代表性的四个模型作为本次评测的基础
evaluator_models = ["gpt-4-1106-preview", "gpt-3.5-turbo-0613", "mixtral-8x7b-instruct-v0.1", "llama-2-70b-chat"]
# 和四个模型同属一个group的模型也应该考虑在内
evaluator_models_familiy = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-0314', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106',
                            'gpt-4-0125-preview', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-1106-preview',
                            'mixtral-8x7b-instruct-v0.1', 'mistral-7b-instruct', 'mistral-7b-instruct-v0.2', 'mistral-medium',
                            'llama-2-70b-chat', 'llama-2-13b-chat', 'llama-2-7b-chat']

evaluator_groups = {}
evaluator_groups["gpt-3.5-turbo-0613"] = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-0314', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106']
evaluator_groups["gpt-4-1106-preview"] = ['gpt-4-0125-preview', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-1106-preview']
evaluator_groups["mixtral-8x7b-instruct-v0.1"] = ['mixtral-8x7b-instruct-v0.1', 'mistral-7b-instruct', 'mistral-7b-instruct-v0.2', 'mistral-medium']
evaluator_groups["llama-2-70b-chat"] = ['llama-2-13b-chat', 'llama-2-70b-chat', 'llama-2-7b-chat']

def balance_length(sub_df, average_and_concat=False):
    verbo_win_df = sub_df[((sub_df["response_a"] > sub_df["response_b"]) & (sub_df["winner_model_a"]==1)) | ((sub_df["response_a"] < sub_df["response_b"]) & (sub_df["winner_model_b"]==1))]
    verbo_los_df = sub_df[((sub_df["response_a"] > sub_df["response_b"]) & (sub_df["winner_model_b"]==1)) | ((sub_df["response_a"] < sub_df["response_b"]) & (sub_df["winner_model_a"]==1))]
    tie_df = sub_df[sub_df["winner_tie"]==1]

    if average_and_concat:
        if len(verbo_win_df) > len(verbo_los_df):
            verbo_win_df = verbo_win_df.sample(len(verbo_los_df))
        elif len(verbo_win_df) < len(verbo_los_df):
            verbo_los_df = verbo_los_df.sample(len(verbo_win_df))

        return pd.concat([verbo_win_df, verbo_los_df, tie_df])
    
    return verbo_win_df, verbo_los_df, tie_df

def reverse_position(sub_df):

    sub_df_rev = copy.deepcopy(sub_df)
    sub_df_rev.loc[:, ['model_a', 'model_b']] = sub_df[['model_b', 'model_a']].values
    sub_df_rev.loc[:, ['response_a', 'response_b']] = sub_df[['response_b', 'response_a']].values
    sub_df_rev.loc[sub_df["winner_model_a"]==1, ["winner_model_a", "winner_model_b"]] = [0, 1]
    sub_df_rev.loc[sub_df["winner_model_b"]==1, ["winner_model_a", "winner_model_b"]] = [1, 0]

    return sub_df_rev

def create_verbo_bias_data(df, sample_num=500):

    new_df = []
    total_len = 0
    for i in range(100):
        new_df.append(df[(abs(df["length_diff"])==i) & (df["response_a"]!=df["response_b"]) & (df["winner_tie"] != 1)])
        total_len += len(new_df[-1])
        if total_len >= sample_num:
            break

    new_df = pd.concat(new_df)

    # new_df_rev = reverse_position(new_df)
    # new_df = pd.concat([new_df, new_df_rev])

    new_df.to_csv(f'test_data/same_verbo.csv', index=False)

    return new_df

if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    def subtract_length(x):
        return len(x.response_a) - len(x.response_b)
    
    df["length_diff"] = df.apply(subtract_length, axis=1)

    # 确保在verbo-bias和position-bias数据中，不包含evaluator模型及其同组模型相关的数据，从而去除self-bias的影响
    df = df[(~df["model_a"].isin(evaluator_models_familiy)) & (~df["model_b"].isin(evaluator_models_familiy))]

    verbo_bias = create_verbo_bias_data(df)