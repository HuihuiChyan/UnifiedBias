# UnifiedBias

本项目致力于衡量大模型评估过程中的bias与logprobs之间的关联，为此构建了包含三种bias的数据（verbosity bias，position bias, self bias)。

但是实验过程中发现，bias的本质其实是预测不准确，将bias分为上述三种类型，是一个不合理、不全面、而且互相包含的实验设置，因此本项目并没有继续进行。

## Usage
从[这个比赛](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k)下载训练集，然后运行`create_test.py`或者`create_new_test.py`来构建测试集。

在这之后，运行`evaluate_judge.py`来调用大模型在meta-evaluation set上进行测试，测试结果可以反映不同大模型作为评估器时，不同类型bias的大小。

在这之后，运行`cal_ans_reliability.py`来计算大模型对于输出答案部分的logprobs，可以对比logprobs和评估结果，观察评估结果和logprobs大小之间的关联。
