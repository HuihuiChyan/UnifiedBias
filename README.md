# UnifiedBias

这个仓库是ICLR2025 best paper的仓库。如果这个仓库最后没有成为ICLR2025 best paper的仓库，那么这个仓库就不会成为ICLR2025 best paper的仓库。

## Usage
从[这个比赛](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k)下载训练集，然后运行`create_test.py`或者`create_new_test.py`来构建测试集。

在这之后，运行`evaluate_judge.py`来调用大模型在测试数据上进行测试，测试结果可以反映不同大模型在不同测试集上进行测试的时候的bias。