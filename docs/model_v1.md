# DualBranchMINet：MI-EEG 双分支解码网络实现规格

**用途：** 作为 Codex 实现 PyTorch 项目的技术规格。  
**数据集：** BCI Competition IV 2a  
**任务：** 被试内四分类运动想象 EEG 解码  
**实验协议：** Session 1 训练/验证，Session 2 测试  
**采样率：** 250 Hz  
**EEG 通道数：** 22  
**模型输出：** 4 类 logits

> 公式统一使用纯文本和代码块表示，不依赖 LaTeX 渲染。张量形状中的 `B` 表示 batch size，`T` 表示时间采样点数，`N` 表示滑窗数量。

---

## 1. 总体架构

```text
同一试次 MI-EEG
│
├── 频带-时间分支
│   ├── 约束可学习 Sinc 滤波器组（5 个频带）
│   ├── 滑窗功率估计：log(mean(x^2) + eps)
│   ├── 频带 × 电极展开为 110 个特征通道
│   ├── DCT 改进 PSCA（仅将 GAP 替换为分组 DCT）
│   ├── 线性投影 + 可学习时间位置编码
│   └── 时间 Transformer
│       └── 时间 token：[B, 16, 64]
│
├── 空间分支
│   ├── 共享时间卷积进行电极特征编码
│   ├── 空间先验约束的动态图 GAT
│   ├── 带空间先验的自适应脑区软分配（3 个区域）
│   └── 区域级 Transformer
│       └── 区域 token：[B, 3, 64]
│
├── 双向 Cross-Attention
│   ├── 时间 Query，区域 Key/Value
│   └── 区域 Query，时间 Key/Value
│
├── 时间注意力池化 + 区域注意力池化
├── 门控特征融合
└── 四分类器
    └── logits：[B, 4]
```

两个分支输入来自同一试次，目标时间窗口一致。时间 token 数量与区域 token 数量可以不同。

## 2. 数据与实验协议

### 2.1 数据集

BCI Competition IV 2a：
- 9 名被试；
- 22 个 EEG 通道；
- 采样率 250 Hz；
- 四分类：左手、右手、双脚、舌头运动想象。

每名被试单独训练一个模型，不将不同被试的数据混合训练。

### 2.2 Session 划分

- Session 1：训练集和验证集。
- Session 2：独立测试集。
- 验证集从 Session 1 内按类别分层划分。
- Session 2 不参与训练、早停、超参数选择或模型选择。

Session 1 的验证比例通过配置项 `val_ratio` 控制，初始值设为 `0.2`。

### 2.3 试次截取

以每个试次的 MI Cue 为时间零点，截取 Cue 后 0.5 s 至 4.5 s 的信号。

```text
t_start = t_Cue + 0.5 seconds
t_end   = t_Cue + 4.5 seconds
epoch_duration = 4 seconds
T = 4 * 250 = 1000 samples
```

必须根据数据文件中的事件标记定位 Cue，不得通过固定试次数组位置猜测事件时间。

### 2.4 通道顺序

默认 22 通道名称配置：

```python
CHANNEL_NAMES = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
```

读取数据后按通道名称统一顺序。电极坐标、邻接矩阵和 EEG 张量必须使用相同通道顺序。

### 2.5 两个分支的输入预处理

**空间分支：**
- 使用原始 EEG；
- 仅执行工频陷波；
- 不额外执行 4–40 Hz 或 8–30 Hz 带通。

**频带-时间分支：**
- 输入原始 EEG；
- 由模型内部的可学习 Sinc 滤波器组完成频带分解；
- 不使用固定 Butterworth 频带滤波替代 Sinc 滤波器组。

Dataset 应分别返回两个分支所需的输入。两者来自同一试次且时间对齐。

### 2.6 Dataset 输出接口

```python
{
    "x_spatial": Tensor[22, 1000],  # 陷波后的空间分支输入
    "x_frequency": Tensor[22, T_context],
    "label": Tensor[],
    "subject": str,
    "session": str,
}
```

`T_context` 由 Sinc 滤波上下文配置决定。若不使用额外上下文，则 `T_context=1000`。若使用额外上下文，则 Dataset 应在目标试次前后读取相应 EEG 片段。

DataLoader 后：
- `x_spatial`: `[B, 22, 1000]`
- `x_frequency`: `[B, 22, T_context]`
- `label`: `[B]`，类别索引统一为 `0, 1, 2, 3`。

---

## 3. 频带-时间分支

### 3.1 约束可学习 Sinc 滤波器组

使用 5 个带通 Sinc 滤波器，初始频带为：

| 滤波器 | 初始低截止频率 | 初始高截止频率 |
|---|---:|---:|
| 1 | 4 Hz | 8 Hz |
| 2 | 8 Hz | 13 Hz |
| 3 | 13 Hz | 20 Hz |
| 4 | 20 Hz | 30 Hz |
| 5 | 30 Hz | 40 Hz |

截止频率通过训练学习，同时保持合法频率范围、频带顺序和正带宽。五个频带按连续频带组初始化。

建议使用共享边界参数化：六个边界初始化为 `[4, 8, 13, 20, 30, 40]`，相邻滤波器共享边界，保证频带顺序。边界参数化和最小带宽通过配置控制。

带通 Sinc 核由低通 Sinc 核相减构造。对称有限长度卷积核乘窗函数后应用于 EEG。滤波器参数在每次 forward 时由当前截止频率重新生成，使截止频率能够通过反向传播更新。

滤波器组对 22 个电极使用相同的频带响应。输入和输出形状：

```text
Input:  [B, 22, T_context]
Output: [B, 5, 22, T_context]
```

滤波后根据上下文配置裁剪出目标 4 s 试次，再进行滑窗功率估计。

**配置项：**
- `num_bands=5`
- `initial_edges_hz=[4, 8, 13, 20, 30, 40]`
- `learnable_cutoffs=true`
- `kernel_size`：尚未最终确定。初始运行值暂设 `251` taps，标记为 provisional。
- `window_function`：初始值 `hamming`，作为配置项。
- `context_samples`：尚未最终确定。暂设 `0` 以便完成 shape test；后续可设置为正数，使 Dataset 提供试次前后的上下文，滤波后裁剪中心 1000 点。
- `share_across_electrodes=true`

`kernel_size` 必须为奇数。Sinc 滤波器应使用 PyTorch 张量运算生成卷积核，确保截止频率参数可训练。

### 3.2 滑窗功率估计

输入形状：

```text
X_sinc: [B, 5, 22, 1000]
```

使用 1 s 窗长、0.2 s 步长：
- `window_size = 250`
- `stride = 50`
- `eps = 1e-6`

对每个频带、每个电极、每个窗口计算对数平均功率：

```text
P[b, f, c, n] =
    log(
        (1 / L) * sum_{t=0}^{L-1} X[b, f, c, n*S + t]^2
        + eps
    )
```

其中：
- `L = 250`，即窗口长度；
- `S = 50`，即滑动步长；
- `n` 表示滑窗索引。

不进行额外 padding 时，时间窗口数：

```text
N = floor((1000 - 250) / 50) + 1 = 16
```

输出形状：

```text
P: [B, 5, 22, 16]
```

### 3.3 频带-电极联合通道

将频带维和电极维合并，采用 band-major 顺序：

```text
j = f * 22 + c
U = reshape(P, [B, 110, 16])
```

每个通道对应一个特定频带与一个特定电极的组合，共 `5 * 22 = 110` 个通道。

### 3.4 DCT 改进 PSCA

使用 AMFTCNet 中的 Product-Sum Channel Attention（PSCA）作为基础结构。改进点只有一个：**将 PSCA 中的 GAP 描述操作替换为沿时间维的分组 DCT 描述操作。**

PSCA 其余结构保持原图中的通道权重生成、乘积和求和路径，不增加 GAP 分支，不增加额外频带注意力。

#### 3.4.1 DCT 描述子

输入形状：

```text
U: [B, 110, 16]
```

沿最后一个维度（16 个滑窗时间位置）执行一维 DCT-II。

采用 FcaNet 风格的分组方式：
- 将 110 个联合通道划分为 3 组；
- 三组分别使用 DCT 阶数 `k=1, 2, 3`；
- 每个通道只计算其所属组对应的一个 DCT 系数；
- 不使用 `k=0`；
- 每个通道最终得到一个标量描述子。

默认使用交错映射：

```python
dct_order = [1, 2, 3]
channel_dct_k[j] = dct_order[j % 3]
```

因此三组通道数分别为 37、37、36。该映射必须做成配置项，便于后续消融和调整。

对通道 `j`，其 DCT 描述子为：

```text
z[b, j] =
    sqrt(2 / N) *
    sum_{n=0}^{N-1} (
        U[b, j, n] *
        cos(pi * k_j * (n + 0.5) / N)
    )
```

其中：
- `N = 16`；
- `k_j` 为该通道分配的 DCT 阶数，取值为 `1、2、3`。

得到：

```text
Z_DCT: [B, 110]
```

DCT 必须使用 PyTorch 张量运算实现，保持对输入特征的梯度传播。

#### 3.4.2 PSCA 通道权重生成与 Product-Sum

将 DCT 描述子送入原 PSCA 的通道权重生成路径，得到 110 个通道权重：

```text
A = ChannelWeight(Z_DCT)
A: [B, 110]
```

将 `A` 扩展为 `[B, 110, 1]`，沿时间维广播。按照用户提供的 AMFTCNet PSCA 结构图，保留乘积与求和路径：

```text
U_PSCA = U * A + A
```

这里的 `A` 在时间维广播，输出形状保持不变：

```text
U_PSCA: [B, 110, 16]
```

PSCA 的通道权重生成网络和 Product-Sum 操作按 AMFTCNet 原模块实现。全连接映射维度、激活函数等参数集中放在 `PSCAConfig` 中。DCT 仅替换原 GAP 描述操作，不增加第二条 GAP 路径。

PSCA 模块接口：

```python
class DCTPSCA(nn.Module):
    def forward(self, x, return_weights=False):
        # x: [B, 110, 16]
        # output: [B, 110, 16]
        # optional weights: [B, 110]
        ...
```

### 3.5 时间特征投影与位置编码

将 PSCA 输出转置：

```text
U_time = transpose(U_PSCA, dims=(1, 2))
U_time: [B, 16, 110]
```

通过线性层映射到 64 维：

```text
H0 = Linear(110, 64)(U_time)
H0: [B, 16, 64]
```

添加可学习时间位置编码：

```text
H1 = H0 + E_pos
E_pos: [1, max_len, 64]
max_len = 128
```

`max_len` 作为配置项。

### 3.6 时间 Transformer

使用 Transformer Encoder：
- `d_model=64`
- `nhead=4`
- `num_layers=2`
- `dim_feedforward=128`
- `dropout=0.3`
- `batch_first=True`
- 激活函数 GELU

输出：

```text
H_T: [B, 16, 64]
```

保留全部时间 token，不在 Cross-Attention 前进行全局池化。

---

## 4. 空间分支

空间分支依次包含共享时间卷积、电极级动态图 GAT、自适应脑区软分配和区域级 Transformer。

### 4.1 共享时间卷积：电极特征编码

输入：

```text
X_spatial: [B, 22, 1000]
```

将每个电极作为一条独立时间序列：

```text
X_reshape: [B * 22, 1, 1000]
```

22 个电极共享同一套时间卷积参数，不在此阶段混合不同电极的信息。

初始卷积结构（所有参数均放在配置文件中）：

```text
Input: [B*22, 1, T]
    |
Conv1d(1, 32, kernel_size=25, stride=5, padding=12)
    |
BatchNorm1d(32) + GELU
    |
Conv1d(32, 64, kernel_size=15, stride=2, padding=7)
    |
BatchNorm1d(64) + GELU
    |
AdaptiveAvgPool1d(1)
    |
[B*22, 64]
```

恢复电极维度：

```text
H_E: [B, 22, 64]
```

### 4.2 空间先验约束的动态图 GAT

#### 4.2.1 KNN 空间图

根据 22 个电极的头皮坐标计算两两距离，构建 KNN 局部邻接图：
- 每个节点连接自身；
- 每个节点连接距离最近的 K 个电极；
- 邻接矩阵对称化；
- 坐标、通道顺序和邻接矩阵保持一致。

`K` 为配置项，初始建议值 `4`，可通过消融调整。

空间图作为注意力 mask，限制 GAT 只在 KNN 邻居之间计算注意力。

#### 4.2.2 动态图注意力

GAT 的注意力权重根据当前试次的电极特征动态计算。

对第 `h` 个注意力头：

```text
h_tilde[i, h] = W[h] * h[i]

e[i, j, h] =
    LeakyReLU(
        a[h]^T * concat(h_tilde[i, h], h_tilde[j, h])
    )
```

对非邻接节点施加 mask，再对邻居维度进行 Softmax：

```text
alpha[i, j, h] = Softmax over j of e[i, j, h]
```

节点更新：

```text
h_new[i] =
    concat over heads h (
        sum over neighbors j (
            alpha[i, j, h] * W[h] * h[j]
        )
    )
```

GAT 层数、注意力头数、残差连接和归一化方式均通过配置控制。初始实现建议 2 层、4 头、输出维度 64。

输出：

```text
H_G: [B, 22, 64]
```

### 4.3 带空间先验的自适应脑区软分配

脑区数量固定为：

```text
R = 3
```

三个区域分别表示：
1. 左侧感觉运动区域；
2. 中线感觉运动区域；
3. 右侧感觉运动区域。

初始空间锚点建议分别取 C3、Cz、C4 的电极坐标，锚点和空间先验带宽均放在配置文件中。

根据电极坐标 `p[c]` 与区域锚点 `mu[r]` 构造空间先验：

```text
P_prior[c, r] =
    Softmax over r (
        - ||p[c] - mu[r]||^2 / (2 * sigma[r]^2)
    )
```

为三个区域设置可学习查询向量 `q[r]`。将电极特征映射为：

```text
u[c] = W_s * h[c]
```

计算动态分配分数：

```text
score[b, c, r] =
    dot(u[b, c], q[r]) / sqrt(64)
    + lambda * log(P_prior[c, r] + eps)
```

对区域维度进行 Softmax：

```text
S[b, c, r] = Softmax over r of score[b, c, r]
S: [B, 22, 3]
```

根据软分配权重聚合区域特征：

```text
Z_region[b, r] =
    sum over c (S[b, c, r] * H_G[b, c])
    /
    (sum over c S[b, c, r] + eps)
```

输出：

```text
Z_region: [B, 3, 64]
```

第一版不增加区域均衡或分配熵正则项。需要通过 `return_aux=True` 导出软分配矩阵。

### 4.4 区域级 Transformer

输入为 3 个区域 token：

```text
Z_region: [B, 3, 64]
```

使用 Transformer Encoder：
- `d_model=64`
- `nhead=4`
- `num_layers=2`
- `dim_feedforward=128`
- `dropout=0.3`
- `batch_first=True`
- 激活函数 GELU

输出：

```text
H_S: [B, 3, 64]
```

---

## 5. 双向 Cross-Attention

输入：
- 时间 token：`H_T [B, 16, 64]`
- 区域 token：`H_S [B, 3, 64]`

实现两个参数独立的 Cross-Attention 模块，并行计算两个方向。

### 5.1 时间查询空间

```text
Q_T = H_T * W_Q_T
K_S = H_S * W_K_S
V_S = H_S * W_V_S

C_T_from_S = MHA(Q_T, K_S, V_S)
H_T_new = CrossBlock(H_T, C_T_from_S)

H_T_new: [B, 16, 64]
```

### 5.2 空间查询时间

```text
Q_S = H_S * W_Q_S
K_T = H_T * W_K_T
V_T = H_T * W_V_T

C_S_from_T = MHA(Q_S, K_T, V_T)
H_S_new = CrossBlock(H_S, C_S_from_T)

H_S_new: [B, 3, 64]
```

每个 `CrossBlock` 使用：
1. Multi-Head Cross-Attention；
2. 残差连接 + LayerNorm；
3. FFN（Linear → GELU → Dropout → Linear）；
4. 残差连接 + LayerNorm。

两个方向均使用交互前的 `H_T` 和 `H_S` 计算 Query、Key、Value，再分别得到更新结果。两个方向参数独立。

### 5.3 Cross-Attention 维度说明

Query 序列长度与 Key/Value 序列长度可以不同。本模型中：
- 时间 Query 长度为 16，空间 Key/Value 长度为 3；
- 空间 Query 长度为 3，时间 Key/Value 长度为 16。

两种方向均可直接进行交叉注意力计算。

---

## 6. 注意力池化与门控融合

### 6.1 双分支注意力池化

分别对交互后的时间 token 和区域 token 做注意力池化。

```text
h_T = AttnPool(H_T_new)
h_T: [B, 64]

h_S = AttnPool(H_S_new)
h_S: [B, 64]
```

注意力池化使用可学习查询向量，对 token 维度计算 Softmax 权重并加权求和。

### 6.2 门控融合

拼接两个分支向量：

```text
h_cat = concat(h_T, h_S)
h_cat: [B, 128]
```

计算逐维门控：

```text
g = Sigmoid(Linear(128, 64)(h_cat))
g: [B, 64]
```

融合：

```text
h_fuse = GELU(
    Linear(128, 64)(
        concat(g * h_T, (1 - g) * h_S)
    )
)
```

融合后使用 LayerNorm 和 Dropout。

输出：

```text
h_fuse: [B, 64]
```

---

## 7. 四分类器

分类头：

```text
Linear(64, 64)
GELU
Dropout(0.3)
Linear(64, 4)
```

输出：

```text
logits: [B, 4]
```

模型 forward 返回 logits。训练时直接使用 `CrossEntropyLoss`，不在模型末尾添加 Softmax。

---

## 8. 归一化与训练配置

### 8.1 归一化

跨 Session 的严格基线建议采用训练集统计量：
- 仅使用 Session 1 训练子集计算均值和标准差；
- 将同一组统计量应用于 Session 1 验证集和 Session 2 测试集；
- 不使用 Session 2 标签或统计结果进行训练和模型选择；
- 不对每个测试试次单独拟合归一化参数。

归一化策略做成配置项。当前建议默认 `train_session_stats`，并分别保存各输入分支需要的统计量。该项仍可调整。

### 8.2 训练配置

| 参数 | 初始值 |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Batch size | 32 |
| Maximum epochs | 200 |
| Weight decay | `1e-4` |
| Dropout | 0.3 |
| Loss | CrossEntropyLoss |
| Early stopping patience | 25–30 |
| Random seeds | 42、2026、2027 |

验证集用于早停和模型选择。Session 2 仅在模型确定后进行最终测试。

第一版不使用数据增强。

### 8.3 评价指标

逐被试保存并报告：
- Accuracy；
- Cohen's Kappa；
- Macro-F1；
- 四分类混淆矩阵。

汇总 9 名被试的均值和标准差。多随机种子实验应保存每个种子的逐被试结果。

---

## 9. 配置项与待确认参数

以下参数必须集中放在 YAML 配置文件中，不要散落在模型代码里。

| 配置项 | 初始值 | 状态 |
|---|---|---|
| `sampling_rate` | 250 | 已确定 |
| `n_channels` | 22 | 已确定 |
| `epoch_start_sec` | 0.5 | 已确定 |
| `epoch_end_sec` | 4.5 | 已确定 |
| `sinc.initial_edges_hz` | `[4,8,13,20,30,40]` | 已确定 |
| `sinc.learnable_cutoffs` | `true` | 已确定 |
| `sinc.kernel_size` | 251 | 临时默认，待确认 |
| `sinc.window_function` | `hamming` | 临时默认 |
| `sinc.context_samples` | 0 | 临时默认，待确认滤波上下文方案 |
| `power.window_size` | 250 | 已确定 |
| `power.stride` | 50 | 已确定 |
| `psca.dct_orders` | `[1,2,3]` | 已确定 |
| `psca.dct_group_mapping` | `channel_index % 3` | 实现默认，可配置 |
| `psca.include_dc` | `false` | 当前按非零阶实现 |
| `spatial.n_regions` | 3 | 已确定 |
| `spatial.region_anchor_names` | `["C3","Cz","C4"]` | 建议初值 |
| `spatial.knn_k` | 4 | 建议初值，待确认 |
| `spatial.gat_layers` | 2 | 建议初值 |
| `spatial.gat_heads` | 4 | 建议初值 |
| `d_model` | 64 | 已确定 |
| `temporal_transformer.layers` | 2 | 已确定 |
| `temporal_transformer.heads` | 4 | 已确定 |
| `region_transformer.layers` | 2 | 已确定 |
| `region_transformer.heads` | 4 | 已确定 |
| `cross_attention.heads` | 4 | 建议初值 |
| `normalization.mode` | `train_session_stats` | 建议基线，待最终确认 |
| `validation.val_ratio` | 0.2 | 建议初值 |

对于标记为“临时默认”或“建议初值”的参数，Codex 应将其集中配置并在 README 中标出，方便修改。不得把这些值伪装成已由用户确定的设计。

---

## 10. PyTorch 项目结构

```text
project/
├── configs/
│   └── default.yaml
├── data/
│   ├── bci2a_reader.py
│   ├── epoching.py
│   ├── preprocessing.py
│   ├── normalization.py
│   ├── dataset.py
│   └── split.py
├── models/
│   ├── sinc_filterbank.py
│   ├── frequency_branch.py
│   ├── dct_psca.py
│   ├── shared_electrode_encoder.py
│   ├── spatial_gat.py
│   ├── adaptive_roi.py
│   ├── region_transformer.py
│   ├── cross_attention.py
│   ├── fusion.py
│   └── dual_branch_minet.py
├── train.py
├── evaluate.py
├── metrics.py
├── tests/
│   ├── test_shapes.py
│   ├── test_sinc_gradients.py
│   ├── test_dct_psca.py
│   └── test_forward_backward.py
└── README.md
```

### 10.1 总模型接口

```python
class DualBranchMINet(nn.Module):
    def forward(
        self,
        x_spatial,
        x_frequency,
        return_aux=False,
    ):
        # x_spatial: [B, 22, 1000]
        # x_frequency: [B, 22, T_context]
        #
        # return:
        #   logits: [B, 4]
        #
        # if return_aux:
        #   return logits, aux
        ...
```

`aux` 至少包含：
- `sinc_cutoffs`
- `power_features`
- `dct_descriptors`
- `psca_weights`
- `time_tokens`
- `gat_attention`
- `roi_assignment`
- `region_tokens`
- `cross_attention_time_to_region`
- `cross_attention_region_to_time`
- `fusion_gate`

### 10.2 模块输入输出

| 模块 | 输入 | 输出 |
|---|---|---|
| SincFilterBank | `[B,22,T_context]` | `[B,5,22,T_context]` |
| SlidingWindowPower | `[B,5,22,1000]` | `[B,5,22,16]` |
| BandElectrodeFlatten | `[B,5,22,16]` | `[B,110,16]` |
| DCTPSCA | `[B,110,16]` | `[B,110,16]` |
| TemporalTransformer | `[B,16,110]` | `[B,16,64]` |
| SharedElectrodeEncoder | `[B,22,1000]` | `[B,22,64]` |
| SpatialPriorGAT | `[B,22,64]` | `[B,22,64]` |
| AdaptiveROI | `[B,22,64]` | `[B,3,64]` |
| RegionTransformer | `[B,3,64]` | `[B,3,64]` |
| BidirectionalCrossAttention | `[B,16,64]`, `[B,3,64]` | `[B,16,64]`, `[B,3,64]` |
| AttentionPooling | `[B,L,64]` | `[B,64]` |
| AdaptiveFusion | two `[B,64]` vectors | `[B,64]` |
| Classifier | `[B,64]` | `[B,4]` |

---

## 11. 实现与测试要求

Codex 按以下顺序实现：

1. BCI 2a 数据读取、通道顺序对齐、Cue 事件定位和试次截取。
2. 两个分支的预处理和 Dataset。
3. Sinc 滤波器组，验证截止频率参数可学习且频带始终合法。
4. 滑窗功率和 DCT 改进 PSCA。
5. 时间 Transformer。
6. 共享时间卷积、KNN 图构建、GAT 和脑区软分配。
7. 区域 Transformer。
8. 双向 Cross-Attention、注意力池化、门控融合和分类头。
9. 逐被试训练、验证、测试及指标保存。

必须提供以下测试：

- Sinc 输出形状正确；
- Sinc 截止频率参数能够获得非零梯度；
- 五个频带的边界顺序和带宽始终合法；
- 滑窗功率输出 `[B,5,22,16]`；
- DCT 描述子输出 `[B,110]`；
- DCTPSCA 输出 `[B,110,16]`；
- 共享时间卷积输出 `[B,22,64]`；
- GAT 输出 `[B,22,64]`；
- ROI 分配矩阵输出 `[B,22,3]`，且每个电极在三个区域上的权重和为 1；
- 区域 token 输出 `[B,3,64]`；
- 双向 Cross-Attention 输出分别为 `[B,16,64]` 和 `[B,3,64]`；
- 最终 logits 输出 `[B,4]`；
- 完成一个 batch 的 forward、loss、backward 和 optimizer step；
- 检查所有中间张量无 NaN/Inf。

---

## 12. 交付内容

Codex 最终应生成：
1. 完整 PyTorch 源码；
2. 默认 YAML 配置；
3. BCI 2a 数据读取和预处理代码；
4. 逐被试训练脚本；
5. 测试与指标计算脚本；
6. 模块 shape 和梯度测试；
7. README，说明数据目录配置、训练命令、测试命令和结果目录。

所有关键结构与超参数必须通过配置文件管理。训练脚本应支持单被试运行，也应支持按 9 名被试批量运行。

## 13. 参考文献

1. Huang, Q., Yang, Y., Li, J., Liu, X., & Liu, X. (2025). *AMFTCNet: A multi-level attention-based multi-scale fusion temporal convolutional network for decoding MI-EEG signals*. Biomedical Signal Processing and Control, 108, 107916. https://doi.org/10.1016/j.bspc.2025.107916
2. Qin, Z., Zhang, P., Wu, F., & Li, X. (2021). *FcaNet: Frequency Channel Attention Networks*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 783–792. https://doi.org/10.1109/ICCV48922.2021.00082
