# MADP: Multi-Head Adversarial Debiasing for Position-Aware CTR Prediction

本项目是论文 **MADP: Multi-Head Adversarial Debiasing for Position-Aware CTR Prediction** 的配套研究代码。MADP 将点击历史按展示位置分组，学习各位置的兴趣，通过共享预测网络输出 20 个位置的 CTR，并使用梯度反转进行位置对抗训练。

## 项目结构

```text
madp-paper/
├── README.md
├── configs/                 # 9 个实验配置及默认配置
├── madp/
│   ├── config.py
│   ├── layers.py           # masked pooling、DIN、Transformer、GRL
│   ├── models.py           # 9 个结构与统一输出接口
│   ├── losses.py           # 观测位置监督和对抗损失
│   ├── data.py             # NPZ 校验与 batch
│   ├── prepare.py          # JSONL 转换
│   ├── synthetic.py        # 合成样本，仅供运行验证
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
├── scripts/run_all.py       # 全部实验与结果汇总
├── scripts/package.py       # 生成独立提交压缩包
└── tests/                  # 结构、梯度、数据和指标测试
```
代码使用 TensorFlow/Keras，独立运行。论文主模型为 `madp_transformer`，同时提供 DIN 版本、消融实验和 PAL/DPIN 对比结构。

本仓库按下列实验定义实现结构；未提供真实数据和论文结果，附带的合成数据仅用于验证运行流程。PAL、DPIN 是统一输入与训练条件下的基线适配，具体范围见 [结构说明](docs/architecture.md)。

## 实验结构

所有实验使用同一份样本、候选特征、历史事件、数据划分和训练协议。默认位置数为 20。

| 配置名 | 对应实验 | 历史处理 | CTR 输出 | 位置对抗 |
| --- | --- | --- | --- | --- |
| `sequence_flat` | 序列直接加入模型，不区分位置 | 全局时间顺序的 embedding 直接展平 | 1 | 无 |
| `position_pool` | 按位置提取历史兴趣，单预测值 | 每位置 masked mean，保留组顺序拼接 | 1 | 无 |
| `transformer_single` | Transformer 理解每组点击历史，单预测值 | 共享组内 Transformer，组兴趣拼接 | 1 | 无 |
| `transformer_multi` | Transformer 保留各位置兴趣，共享 MLP | 共享组内 Transformer | 20 | 无 |
| **`madp_transformer`** | **Transformer + 位置对抗 + 20 位置预测** | **共享组内 Transformer** | **20** | **GRL** |
| `din_single` | DIN 理解每组点击历史，单预测值 | 每组分别进行目标相关的 DIN attention | 1 | 无 |
| `madp_din` | DIN + 位置对抗 + 20 位置预测 | 每组分别进行 DIN attention | 20 | GRL |
| `pal` | PAL 分解替换多位置预测与位置对抗 | 组内 Transformer + 单个 relevance 网络 | 1 个观测 CTR；另提供 relevance | 无 |
| `dpin` | DPIN | context-aware 组内池化，再跨位置 Transformer | 20 | 无 |

这里“20 个头”指同一个 MLP 在 20 个位置表示上共享参数，各自输出一个 CTR；并不是 20 套独立 MLP。多输出模型的第 `k` 列表示候选放在位置 `k` 时的预测。每个曝光样本只用实际展示位置对应的预测计算点击损失。

## 安装与快速运行

建议 Python 3.12。以下命令在本项目根目录运行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# 生成三个独立的合成数据划分，仅验证程序，不代表论文实验。
python -m madp.synthetic --config configs/default.json --output data/demo \
  --train-size 256 --val-size 128 --test-size 128

python -m madp.train --config configs/madp_transformer.json \
  --train data/demo/train.npz --valid data/demo/valid.npz \
  --output runs/madp_transformer --epochs 1

python -m madp.evaluate --run runs/madp_transformer \
  --data data/demo/test.npz --output reports/madp_transformer

python -m pytest -q
```

一次运行全部 9 个对比实验：

```bash
python scripts/run_all.py --config configs/default.json \
  --train data/demo/train.npz --valid data/demo/valid.npz \
  --test data/demo/test.npz --output runs/all --epochs 1
```

脚本生成各模型的验证选优 checkpoint、测试指标和汇总表。真实实验应替换为预先固定的 train/valid/test，使用完整训练轮数，并使用多个随机种子分别运行；不应根据测试集选择模型或调参。

## 数据接口

一个样本是一条曝光记录，点击为 `label=1`，未点击为 `label=0`。历史只包含当前请求发生之前的点击。物品 ID `0` 专用于 padding；合法物品 ID 为 `1..vocab_size-1`。代码和生成报告中的位置均为 **0..19**，对应实际第 **1..20** 位。不要隐式对原始 rank 取模；如果原始数据采用其他位置定义，应在数据准备时明确转换。

NPZ 不包含 Python 对象，只使用下列数值数组：

| 字段 | 形状 | 类型与含义 |
| --- | --- | --- |
| `candidate_id` | `[N]` | 整数，当前候选 ID |
| `context` | `[N, context_dim]` | 浮点数，请求前已知的用户/场景特征；不含当前展示位置或点击标签 |
| `history_ids` | `[N, 20, history_length]` | 整数，按历史展示位置分组，每组从旧到新，左对齐，尾部补 0 |
| `raw_history_ids` | `[N, 20 * history_length]` | 同一批保留的历史点击，按全局时间从旧到新，尾部补 0；不编码历史展示位置 |
| `position` | `[N]` | 整数，实际曝光位置，0..19 |
| `label` | `[N]` | 浮点数，0.0 或 1.0 |

`raw_history_ids` 不能简单按位置展平 `history_ids` 得到，否则无位置基线会通过固定分段偷看历史位置。程序校验两种历史包含相同的事件多重集；时间顺序和“请求前历史”的真实性由上游数据构建保证。

也可以先准备 JSONL，每行如下（`history` 已按全局时间从旧到新排列）：

```json
{"candidate_id": 3, "context": [0.2, 0.1, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0], "position": 0, "label": 1.0, "history": [{"item_id": 8, "position": 2}, {"item_id": 5, "position": 0}]}
```

```bash
python -m madp.prepare --config configs/default.json \
  --input data/train.jsonl --output data/train.npz
```

转换器对每个历史位置保留最近 `history_length` 次点击，分别构建分组序列和原时间顺序序列。词表、上下文归一化等只在训练数据上拟合，再应用到验证与测试集；数据划分建议遵循实际曝光时间。

## 训练目标与推断

- 单输出模型：直接对单个 CTR 计算 BCE。
- 多输出模型：对 `logits[i, position[i]]` 计算 BCE，其他 19 个位置没有该样本的点击标签，不扩展伪标签。
- MADP：`loss = observed_BCE + adversarial_weight * position_CE`。位置判别器预测**当前曝光位置**；共享表示经 GRL 回传反向梯度。显式位置 embedding 仅在 CTR 预测头中使用，不进入判别器。具体表征连接与梯度公式见 [architecture.md](docs/architecture.md)。
- PAL：`p_click = p_examination(position) * p_relevance(features, history)`，对乘积计算观测 BCE；`relevance` 是排序分数，`ctr` 是观测位置点击概率，二者不能混用。这里不固定首位置 examination 为 1，也不强加单调约束。
- DPIN：位置塔只读取请求 context、历史、位置 embedding；候选物品通过独立 base tower 接入共享组合网络。

多输出模型推断保留 `[batch, 20]`，不会默认固定为首位置或平均为一个分数。指定位置的打分可显式调用 `observed_ctr(outputs, position, num_positions=20)`。GRL 分支是训练辅助目标，不代表通过观测数据即可保证因果意义上的无偏 CTR。

## “位置 / 样本量”结果表

每次评估输出总体和全部 20 个位置的样本数、点击数、真实 CTR、AUC、LogLoss。空位置与单标签位置的 AUC 写为 `NA`，不填写虚构分数。加权位置 AUC 只统计 AUC 可定义的位置，并同时报告样本覆盖率。

`scripts/run_all.py` 进一步输出以“位置 / 样本量”为前两列、9 个模型 AUC 为后续列的对比表，另附总体指标。示意如下，具体数值由真实运行生成：

| 位置 | 样本量 | sequence_flat | position_pool | transformer_single | transformer_multi | madp_transformer | din_single | madp_din | pal | dpin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0–19 | 从测试集统计 | 各位置 AUC | 各位置 AUC | 各位置 AUC | 各位置 AUC | 各位置 AUC | 各位置 AUC | 各位置 AUC | 各位置 AUC | 各位置 AUC |

