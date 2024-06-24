import copy
import random
import pandas as pd

df = pd.read_csv("train.csv")

# all_models = sorted(list(set(df["model_a"].to_numpy().tolist())))
# print(all_models) # 共计 64个模型

# 我们选择了比较有代表性的四个模型作为本次评测的基础
evaluator_models = ["gpt-4-1106-preview", "gpt-3.5-turbo-0613", "vicuna-13b", "llama-2-13b-chat"]
# 和四个模型同属一个group的模型也应该考虑在内
evaluator_group_models = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-0314', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106',
                          'gpt-4-0125-preview', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-1106-preview',
                          'vicuna-13b', 'vicuna-33b', 'vicuna-7b',
                          'llama-2-13b-chat', 'llama-2-70b-chat', 'llama-2-7b-chat']

def balance_length(sub_df, average_and_concat=False):
    verbo_win_df = sub_df[(sub_df["response_a"] > sub_df["response_b"]) & (sub_df["winner_model_a"]==1) | (sub_df["response_a"] < sub_df["response_b"]) & (sub_df["winner_model_b"]==1)]
    verbo_los_df = sub_df[(sub_df["response_a"] > sub_df["response_b"]) & (sub_df["winner_model_b"]==1) | (sub_df["response_a"] < sub_df["response_b"]) & (sub_df["winner_model_a"]==1)]
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

def create_self_bias_data(df, sample_num=500):
    self_bias = {}
    for model_name in evaluator_models:
        self_df = df[(df["model_a"]==model_name) | (df["model_b"]==model_name)]
        self_win_df = df[(df["model_a"]==model_name) & (df["winner_model_a"]==1) | (df["model_b"]==model_name) & (df["winner_model_b"]==1)]
        self_los_df = df[(df["model_a"]==model_name) & (df["winner_model_b"]==1) | (df["model_b"]==model_name) & (df["winner_model_a"]==1)]
        self_tie_df = df[((df["model_a"]==model_name) | (df["model_b"]==model_name)) & (df["winner_tie"]==1)]

        # sanity check
        assert len(self_win_df) + len(self_los_df) + len(self_tie_df) == len(self_df)

        # 确保在self-bias数据中，verbo_win和verbo_los数量相同，从而去除verbo-bias的影响
        self_win_df = balance_length(self_win_df, average_and_concat=True)
        self_los_df = balance_length(self_los_df, average_and_concat=True)

        self_win_df = self_win_df.sample(sample_num)
        self_los_df = self_los_df.sample(sample_num)

        self_win_df_rev = reverse_position(self_win_df)
        self_win_df = pd.concat([self_win_df, self_win_df_rev])
        self_los_df_rev = reverse_position(self_los_df)
        self_los_df = pd.concat([self_los_df, self_los_df_rev])

        self_bias[model_name] = {"self_win": self_win_df, "self_los": self_los_df}

        self_win_df.to_csv(f'self_win_{model_name}.csv', index=False)
        self_los_df.to_csv(f'self_los_{model_name}.csv', index=False)
    
    return self_bias

def create_verbo_bias_data(df, sample_num=500):

    verbo_win_df = df[(df["longer"]=="a") & (df["winner_model_a"]==1) | ((df["longer"]=="b") & (df["winner_model_b"]==1))]
    verbo_los_df = df[(df["longer"]=="a") & (df["winner_model_b"]==1) | ((df["longer"]=="b") & (df["winner_model_a"]==1))]

    # sanity check
    assert len(verbo_win_df) + len(verbo_los_df) + len(df[(df["winner_tie"]==1) | (df["longer"]=="tie")]) == len(df)

    verbo_win_df = verbo_win_df.sample(sample_num)
    verbo_los_df = verbo_los_df.sample(sample_num)

    verbo_win_df_rev = reverse_position(verbo_win_df)
    verbo_win_df = pd.concat([verbo_win_df, verbo_win_df_rev])
    verbo_los_df_rev = reverse_position(verbo_los_df)
    verbo_los_df = pd.concat([verbo_los_df, verbo_los_df_rev])

    verbo_bias = {"verbo_win": verbo_win_df, "verbo_los": verbo_los_df}

    verbo_win_df.to_csv(f'verbo_win.csv', index=False)
    verbo_los_df.to_csv(f'verbo_los.csv', index=False)

    return verbo_bias

def create_position_bias_data(df, sample_num=500):

    left_win_df = df[df["winner_model_a"]==1]
    left_los_df = df[df["winner_model_b"]==1]

    # 确保在position-bias数据中，verbo_win和verbo_los数量相同，从而去除verbo-bias的影响
    left_win_df = balance_length(left_win_df, average_and_concat=True)
    left_los_df = balance_length(left_los_df, average_and_concat=True)

    left_win_df = left_win_df.sample(sample_num)
    left_los_df = left_los_df.sample(sample_num)

    left_win_df_rev = reverse_position(left_win_df)
    left_los_df_rev = reverse_position(left_los_df)
    left_win_df = pd.concat([left_win_df, left_los_df_rev])
    left_los_df = pd.concat([left_los_df, left_win_df_rev])

    position_bias = {"left_win": left_win_df, "left_los": left_los_df}

    left_win_df.to_csv(f'left_win.csv', index=False)
    left_los_df.to_csv(f'left_los.csv', index=False)

    return position_bias


if __name__ == "__main__":

    random.seed(42)

    def compare_length(x):
        if len(x.response_a) > len(x.response_b):
            return "a"
        elif len(x.response_a) < len(x.response_b):
            return "b"
        else:
            return "tie"
            
    df["longer"] = df.apply(compare_length, axis=1)

    self_bias = create_self_bias_data(df)

    # 确保在verbo-bias和position-bias数据中，不包含evaluator模型相关的数据，从而去除self-bias的影响
    df = df[(~df["model_a"].isin(evaluator_group_models)) & (~df["model_b"].isin(evaluator_group_models))]

    verbo_bias = create_verbo_bias_data(df)
    position_bias = create_position_bias_data(df)