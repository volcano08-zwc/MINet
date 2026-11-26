# Session 1 全量训练与最高测试 Acc 选模

新增的三个 v1 改进实验统一保留频移增强，使用 `run_v1_experiments.py` 串行执行。完整实验清单与命令见 [v1 实验记录](docs/v1_experiments.md)。

统一使用主入口 `train_2a.py`，默认协议为 `--protocol full-session1`。使用 Session 1 全部保留试次拟合归一化统计量并训练。每轮训练后对 Session 2 执行 eval 前向，执行完指定 epoch 数后报告最高测试 Accuracy。相同最高值出现多次时取最早轮次。

此实验使用测试集选择模型，报告口径为“200 轮中的最高测试 Acc”。论文和实验对比需明确该口径，存在测试集选模带来的乐观偏差。

在服务器项目目录激活 CUDA 环境后执行以下命令，使用物理 GPU 3。

```bash
cd /home/jqsj/hzy/zwc/model/DualBranchMINet
CUDA_VISIBLE_DEVICES=3 python -u train_2a.py \
  --subjects all --seeds 42 --epochs 200 --device cuda:0 \
  --run-version v1 --output-dir output/v1_full_session1_seed42
```

`CUDA_VISIBLE_DEVICES=3` 将物理 GPU 3 映射到当前进程的 `cuda:0`。`--epochs` 默认 200，`--subjects` 支持单被试编号或 `all`，`--seeds` 支持显式指定 42。数据和模型超参数从 `config/bci2a.yaml` 读取，支持 `--data-dir`、`--labels-dir` 和 `--config` 覆盖。

本轮默认版本标记为 `v1`，可通过 `--run-version` 设置。结果目录默认包含版本标记。配置、历史记录、检查点、逐试次预测、指标 JSON、汇总 JSON/CSV、曲线标题和日志均记录该版本。

每名被试运行全部指定轮次，使用 Session 1 全量训练，无验证划分和早停。按最高测试 Acc 选中的模型保存在 `best_model.pt`，对应的 epoch、Acc、Loss、Kappa、Macro-F1、混淆矩阵和预测分别写入 `test_metrics.json` 与 `test_predictions.npz`。

`history.json`、`history.csv`、`accuracy_curve` 和 `loss_curve` 保存逐 epoch 的训练与测试记录。曲线提供 PNG、SVG，标注最高测试 Acc 所在轮次。混淆矩阵对应所选模型。`split.json` 与结果文件明确记录测试集参与模型选择。

服务器需同步最新 `train_2a.py`、`data/augmentation.py` 和 `plot_results.py`，并保留配套的 `model/`、`data/bci2a.py`、`config/`、`metrics.py`、`utils.py`。完整项目同步也可。曲线在训练结束时自动生成。已有 `plot_results.py --run-dir` 重绘命令面向 v1 的训练/验证历史。旧 `train_full_session1.py` 是转发主入口的兼容文件。

## 命令选择增强

两种实现都放在 `data/augmentation.py`，通过 `--augmentation none|sr|frequency_shift` 选择，默认 `none`。模型结构、DCT 阶数和输入维度沿用当前配置。

S&R 默认每个原始 batch 追加同样数量的增强试次，四类尽量均衡抽样。每条试次默认分为 8 段，每段 125 点，同类别、相同时间位置抽取，全部电极共用片段来源。原始 batch 32 时实际训练输入 64 条，288 条原始试次对应每轮 576 条训练呈现、9 次更新。尾批次按实际原始 batch 数量追加。`--sr-ratio 3` 表示每条原始样本追加三条增强样本，总呈现量为四倍。

```bash
CUDA_VISIBLE_DEVICES=3 python -u train_2a.py \
  --subjects all --seeds 42 --epochs 200 --device cuda:0 --run-version v1 \
  --augmentation sr --sr-segments 8 --sr-ratio 1 \
  --output-dir output/v1_sr_seed42
```

频移默认以 50% 概率对每个训练试次施加 `[-0.5,+0.5] Hz` 的随机平移，一个试次全部电极共用同一平移量。每轮样本数量与更新次数等同于原始输入，每次读取重新抽样。

```bash
CUDA_VISIBLE_DEVICES=3 python -u train_2a.py \
  --subjects all --seeds 42 --epochs 200 --device cuda:0 --run-version v1 \
  --augmentation frequency_shift --frequency-probability 0.5 --frequency-max-hz 0.5 \
  --output-dir output/v1_frequency_shift_seed42
```

增强在原始频率输入上完成，空间输入由同一增强信号的中心目标窗口执行既有陷波生成。随后两路使用干净 Session 1 拟合的统计量分别归一化。测试数据经过原有预处理与归一化。增强方法和强度写入配置、检查点及结果 JSON，日志与图标题标注增强方法。训练 Acc、Loss 按实际参与训练的原始和增强样本共同统计。

选择 `--protocol validation` 时也支持这两种增强，S&R 片段池和归一化均只使用分层划分后的训练子集，验证和测试沿用各自原始试次。
