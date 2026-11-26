# v1 实验记录与串行运行说明

本项目的初始基线、训练协议调整、S&R、频移以及本次正则化、轻量化、GroupNorm 对照实验，均归属 **v1**。`version=v1` 表示模型系列，`experiment_variant` 区分该系列中的不同实验，不另命名为 v2。

本文记录实验设计、配置、运行状态与本地结果位置。准确率数值、逐轮曲线、预测、混淆矩阵、日志和权重仅保存在 `output/`，不再向 Git 添加或更新结果快照。

## 实验清单

“已完成”依据当前本地已传回的九名被试结果。新增三组完成实现和本地功能验证，尚未在服务器完成正式训练。

| 编号 | 实验 | 与初始基线的主要区别 | 增强 | 状态 | 本地结果目录 |
| --- | --- | --- | --- | --- | --- |
| V1-00 | 初始验证集选模 | Session 1 内部划分 20% 验证集，按验证 Loss 选模，早停 | 无 | 已完成 | `output/v1_seed42/` |
| V1-01 | 全量 Session 1 | 不划分验证集、不早停，200 轮，每轮测试，按最高测试 Acc 选模 | 无 | 已完成 | `output/v1_full_session1_seed42/` |
| V1-02 | S&R | 全量训练协议；同类、同时间位置、所有电极同步拼接；8 段，追加比例 1 | S&R | 已完成 | `output/v1_sr_seed42/` |
| V1-03 | 频移对照 | 全量训练协议；每条训练样本以 50% 概率同步平移所有电极频率，范围 ±0.5 Hz | 频移 | 已完成，作为后续共同对照 | `output/v1_frequency_shift_seed42/` |
| V1-04 | `regularized` | 在频移对照上加强权重衰减、Dropout，加入训练标签平滑 | 频移 | 已实现，待服务器训练 | `output/v1_improvements_seed42/v1_regularized/` |
| V1-05 | `lightweight` | 在频移对照上缩小模型宽度、注意力与 GAT 层数 | 频移 | 已实现，待服务器训练 | `output/v1_improvements_seed42/v1_lightweight/` |
| V1-06 | `groupnorm` | 在频移对照上将电极编码器的两处 BatchNorm 换为 GroupNorm | 频移 | 已实现，待服务器训练 | `output/v1_improvements_seed42/v1_groupnorm/` |

三个新增方案互为独立对照，分别从基础 YAML 配置构建，不叠加其他方案的改动。S&R 代码保留以便复现实验，串行脚本仅使用频移。

## 三组实验共同设置

- BCI2a 全部九名被试分别训练，随机种子 42，每名被试 200 个 epoch，每组共 9 次独立训练。
- Session 1 全部试次用于训练；Session 2 每轮仅做评估。无验证集，无早停；保留测试 Acc 最高轮次，同分时保留最早轮次。
- 学习率 `3e-4`，原始 batch size 32，AdamW；频移不增加 batch 样本数。
- 频移概率 `0.5`，频移范围 `[-0.5,+0.5] Hz`，同一试次所有电极使用相同偏移。验证及测试不增强。
- 保留当前预处理，包括 cue 后 `[0.5,4.5)` 秒、空间分支 50 Hz 陷波、训练会话逐通道统计标准化；本次不同时引入新的时间窗、带通或 EA。
- DCT 阶数仍为 `[1,2,3]`，Sinc 五频带、功率窗 250 点、步长 50 点、ROI 三个区域均保留。
- 测试集用于选模的事实写入协议元数据。报告的是各被试 200 轮内的最高测试 Acc；队列汇总对此求被试平均，不将所有被试的最佳轮次强行统一。

## 具体配置

实验预设集中在 `model/bci2a.py` 的 `apply_v1_experiment()`。GroupNorm 的实际结构开关位于 `SharedElectrodeEncoder`，训练标签平滑设置由模型读取并在训练入口中应用。

| 设置 | 频移对照 | `regularized` | `lightweight` | `groupnorm` |
| --- | --- | --- | --- | --- |
| 权重衰减 | `1e-4` | **`1e-3`** | `1e-4` | `1e-4` |
| 全局、GAT 注意力、融合、分类器 Dropout | `0.3` | **`0.4`** | `0.3` | `0.3` |
| 训练标签平滑 | `0` | **`0.1`** | `0` | `0` |
| 特征宽度 `d_model` | 64 | 64 | **32** | 64 |
| 时间 / 区域 Transformer 层数 | 2 / 2 | 2 / 2 | **1 / 1** | 2 / 2 |
| 时间 / 区域 / 交叉注意力 FFN 隐藏维度 | 128 | 128 | **64** | 128 |
| 共享电极卷积通道 | `[32,64]` | `[32,64]` | **`[16,32]`** | `[32,64]` |
| GAT 层数 | 2 | 2 | **1** | 2 |
| 分类器隐藏维度 | 64 | 64 | **32** | 64 |
| 电极卷积归一化 | BatchNorm | BatchNorm | BatchNorm | **GroupNorm，8 组** |
| 可训练参数量 | 288,264 | 288,264 | **63,912** | 288,264 |

`regularized` 训练 Loss 为标签平滑交叉熵；三组的验证、测试 Loss 均为未平滑的交叉熵。这两种目标下的训练和测试 Loss 数值不宜直接作为同一个目标比较。

三组最终输出均为 `[B,4]`。轻量版的时间 token 为 `[B,16,32]`、区域 token 为 `[B,3,32]`、融合特征为 `[B,32]`。原始架构图仍对应宽度 64 的基线。

## Linux 服务器串行运行

先将更新后的项目代码同步到服务器，包括 `model/`、`config/`、`train_2a.py`、`run_v1_experiments.py`、`evaluate.py` 以及现有数据和绘图依赖。

```bash
screen -S minet_v1_improvements
conda activate dcanet
cd /home/jqsj/hzy/zwc/model/DualBranchMINet
python -u run_v1_experiments.py --gpu 0
```

该命令使用物理 GPU 0，将它映射为子进程中的 `cuda:0`。依次完成正则化组全部九名被试，再开始轻量化组，最后运行 GroupNorm 组。每次只运行一个训练进程。要使用物理 GPU 3，将最后一行改为 `--gpu 3`。

也可继承已有的 GPU 可见性设置：

```bash
CUDA_VISIBLE_DEVICES=3 python -u run_v1_experiments.py
```

`Ctrl+A` 后按 `D` 脱离 Screen，训练继续；使用 `screen -r minet_v1_improvements` 返回会话。

先查看计划而不训练：

```bash
python run_v1_experiments.py --gpu 0 --dry-run
```

覆盖数据位置、随机种子、训练轮数或输出根目录：

```bash
python -u run_v1_experiments.py --gpu 0 --seed 42 --epochs 200 --subjects all --output-dir output/v1_improvements_seed42_run02
```

支持 `--data-dir`、`--labels-dir` 和 `--config`。默认 YAML 已设置服务器数据路径。重新运行时应选择新的输出目录，脚本不会覆盖已有实验目录、日志或队列清单。

如只补跑未完成的方案，可在新的输出根目录指定，例如：

```bash
python -u run_v1_experiments.py --gpu 0 --experiments lightweight groupnorm --output-dir output/v1_improvements_seed42_retry
```

脚本没有断点续训功能。某组报错或被中断后停止队列，后续组不会自动开始；先查看该组日志，修复问题后选择新目录重新运行所需方案。

## 单独运行一组

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_2a.py --experiment regularized --subjects all --seeds 42 --epochs 200 --device cuda:0 --run-version v1 --output-dir output/v1_regularized_frequency_shift_seed42
```

`--experiment regularized|lightweight|groupnorm` 默认启用频移，可通过原有频移参数调整强度。`--experiment baseline` 保留原训练入口的行为；显式 `--augmentation` 可覆盖增强方式，用于后续独立对照。

## 运行产物与记录规则

默认 `output/v1_improvements_seed42/` 中包含：

```text
run_manifest.json                 # 实际命令、GPU、v1 标记、顺序与完成状态
experiment_comparison.csv          # 已完成实验的平均最高测试 Acc 与参数量
experiment_comparison.json
logs/
  v1_regularized.log
  v1_lightweight.log
  v1_groupnorm.log
v1_regularized/A01/seed_42/...
v1_lightweight/A01/seed_42/...
v1_groupnorm/A01/seed_42/...
```

每个被试仍保存完整配置、归一化统计、模型检查点、最高轮次的预测和指标、逐轮历史、Acc/Loss 曲线、计数及归一化混淆矩阵。配置、日志、历史、检查点协议及结果中记录 `v1` 和具体实验名，复评时从检查点配置自动重建相应结构。

每完成一组，队列更新本地比较表。只有确认该组所有被试完成指定轮数且结果设置匹配，才启动下一组。所有运行产物留在忽略目录，后续提交仅包含代码、配置、测试和本说明。
