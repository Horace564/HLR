# 间隔重复学习中的回忆率预测

[English](README.md) · 简体中文

本项目研究模型结构、预测目标和学习历史特征如何影响回忆率预测。基于 Duolingo 学习记录，统一比较十二个模型，并进一步分析不同评价指标和模型输出背后的差异。

[最终报告](deliverables/ELEC3544_Group4_final_report.pdf) · [演讲材料](deliverables/final_presentation.pptx) · [Notebook 导览](NOTEBOOKS.md)

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 结果与发现](#2-结果与发现)
- [3. 快速开始](#3-快速开始)
- [4. 项目结构](#4-项目结构)
- [5. 参考文献与成员](#5-参考文献与成员)

## 1. 项目概述

### 1.1 背景与动机

间隔重复通过安排复习时间帮助学习者长期保持记忆。预测学习者再次遇到某个词元时的回忆率，可以为复习时间安排提供依据。Half-Life Regression（HLR）使用可学习的记忆半衰期描述遗忘过程，再结合距上次练习的时间预测回忆率。

受到 HLR 及后续神经遗忘曲线模型的启发，本项目研究用户与语言特征、非线性模型和不同预测目标对预测表现的影响。十二个模型使用相同的清洗数据和固定的评价切分。

### 1.2 数据集

数据来自 Settles 与 Meeder（2016）发布的 Duolingo 学习记录。每行表示一名学习者针对一个词元的一次练习 session。预测目标 `p_recall` 是该 session 内的正确回答比例，计算方式为 `session_correct / session_seen`。

原始数据包含 **12,854,226 条记录、12 个字段**，涵盖历史练习表现（`history_seen`、`history_correct`）、距上次练习的时间（`delta`）、学习者与词元标识、学习语言与界面语言、时间戳，以及本次 session 的作答结果。

| 数据检查 | 结果 |
| --- | --- |
| 完全重复记录 | 删除 81 条，保留首次出现的记录 |
| 清洗后记录数 | 12,854,145 |
| 去重后的基础有效性异常 | 0 |
| 历史计数不一致记录 | 保留并报告 101,466 条 |

清洗后的数据按原文件行顺序划分为 **Train 72%（9,254,984 条）**、**Validation 18%（2,313,746 条）**和 **Test 10%（1,285,415 条）**。Validation 用于选参和提前停止，Test 用于最终评价。标签分布高度集中在接近完全回忆的位置。

### 1.3 建模方式

基础输入为 `[sqrt(1 + c), sqrt(1 + w), 1]`，其中 `c = history_correct`，`w = history_seen - history_correct`。复习间隔从秒换算为天，`Δ = delta / 86400`。

回忆率标签统一裁剪到 `[0.0001, 0.9999]`。半衰期模型使用由标签反推的目标 `h = clip(-Δ / log2(p), 15/1440, 274)` 天，再通过遗忘曲线预测回忆率

$$
\hat p = 2^{-\Delta/\hat h}.
$$

预测值使用相同的裁剪边界。半衰期目标由回忆率标签计算得到。模型分为以下几组。

| 模型类别 | 模型 | 预测方式 |
| --- | --- | --- |
| 基线 | HLR、HLR -lex、Train mean | 线性估计半衰期，或预测训练集回忆率均值 |
| HLR 特征扩展 | HLR-User、HLR-Lang、HLR-User+Lang | 在 HLR 中加入用户和/或语言 one-hot 特征 |
| 神经 HLR | N-HLR、N-HLR -lex | 四个 ReLU 隐藏单元预测 `log2(h)`，再代入遗忘曲线 |
| 树模型 | RF-h、RF-p、LGBM-h、LGBM-p | 根据历史预测 `log2(h)`，或根据历史与 `Δ` 直接预测回忆率 |

`-lex` 表示去除词元特征。Notebook 中的 RF-h 与 LGBM-h 分别命名为 `RF-logh (E1)` 和 `LGBM-logh (E3)`，对应的直接回忆率预测模型为 E2 和 E4。

输入使用当前 session 之前的信息，类别词表只从 Train 学习。时间戳核验确认所有 Train 记录均早于所有 Test 记录。主要指标为 MAE、RMSE 和 AUC，其中 AUC 的二值标签由 `np.rint(p_target)` 得到。

## 2. 结果与发现

### 2.1 五点主要观察

下表沿用最终报告在相同 **1,285,415 条 Test 记录**上的结果。MAE、RMSE 越低越好，AUC 越高越好。加粗表示该列的最佳结果。

| 模型 | MAE ↓ | RMSE ↓ | AUC ↑ |
| --- | ---: | ---: | ---: |
| HLR | 0.128033 | 0.290877 | 0.537856 |
| HLR -lex | 0.128283 | 0.290873 | 0.537459 |
| Train mean | 0.174210 | 0.271138 | 0.500000 |
| HLR-User | 0.127069 | 0.290409 | 0.540654 |
| HLR-Lang | 0.127938 | 0.290823 | 0.537993 |
| HLR-User+Lang | 0.126819 | 0.290242 | 0.540222 |
| N-HLR | **0.118898** | 0.287320 | 0.539874 |
| N-HLR -lex | 0.118909 | 0.287335 | 0.540314 |
| RF-h | 0.164276 | 0.317893 | 0.549080 |
| RF-p | 0.170206 | **0.267520** | 0.604966 |
| LGBM-h | 0.164277 | 0.317891 | 0.549281 |
| LGBM-p | 0.170254 | 0.267539 | **0.605412** |

1. **用户与语言特征只带来小幅改善。** HLR-User+Lang 将 MAE 从 0.128033 降至 0.126819，单独加入一类特征的变化更小。
2. **神经 HLR 的 MAE 最低。** N-HLR 达到 0.118898，去除词元后的版本在该指标上几乎相同。
3. **相同预测目标下，RF 与 LightGBM 表现高度接近。** 半衰期版本和直接回忆率版本分别呈现非常接近的总体指标。
4. **树模型的预测方式影响指标取舍。** 相比半衰期版本，直接预测回忆率可以降低 RMSE、提高 AUC，但 MAE 更高。这一版本同时增加了 `Δ` 输入，因此比较涉及目标与特征两方面的变化。
5. **模型排序取决于评价指标。** N-HLR 的 MAE 最低，RF-p 的 RMSE 最低，LGBM-p 的 AUC 最高，适合的模型取决于评价目标。

### 2.2 进一步分析

**N-HLR 的改善是否来自非线性？** 对照实验比较原始 HLR -lex、使用神经模型训练方式的线性 HLR，以及四单元 ReLU N-HLR。所有模型统一使用不含词元的输入，两种对齐模型使用三个配对随机种子。

| 模型 | Test MAE |
| --- | ---: |
| Original HLR -lex | 0.128283 |
| Aligned-Linear-HLR -lex | 0.119907 ± 0.000016 |
| N-HLR -lex | 0.118903 ± 0.000053 |

大部分改善出现在损失和优化方式对齐后。非线性另带来平均 **0.001004** 的 MAE 改善，三个种子的方向一致。即使采用相同的 L2 系数，两种参数化对应的函数级正则约束仍有差别，这是实验的限制。详细实验见 [08：非线性消融](08_nonlinearity_ablation.ipynb)。

**为什么 MAE 和 RMSE 给出不同排序？** Test 中 83.603% 的标签位于裁剪上界。HLR 在这个大样本组上误差更小，有利于降低 MAE，但对低回忆率记录可能严重高估。平方会提高大误差相对于小误差的贡献，因此这些记录更能影响 MSE 及其平方根 RMSE。按组计算 MAE/MSE contribution，可以解释 HLR 与 RF-p 的排名反转。详细分析见 [09：MAE 与 RMSE 分析](09_mae_rmse_analysis.ipynb)。

**为什么 RF 与 LightGBM 如此相似？** 我们首先检查了模型复用、行索引错位、缓存覆盖和后处理错误。分别重建预测，并比较原始输出与裁剪后的输出，排除了所检验的实现问题。一个合理解释是两种方法在相同输入与目标下学到了高度相似的预测模式，裁剪进一步提高了半衰期版本的表面相似性。这里仅进行了初步诊断，更系统的解释留作 future work。详细分析见 [10：RF 与 LightGBM 相似性](10_rf_lightgbm_similarity.ipynb)。

以上发现基于当前离线切分。历史计数异常被保留，同一学习者也可能出现在 Train 与 Test 中。后续可以扩展按学习者隔离的评价，并改变特征和随机种子进行验证。

## 3. 快速开始

### 3.1 环境准备

使用 **64 位 Python 3.11**。实验在 CPU 上运行。全量随机森林训练需要较多内存，数据处理与模型缓存会占用数 GB 磁盘空间，耗时取决于硬件和已有缓存。

克隆仓库并进入项目根目录

```bash
git clone https://github.com/Horace564/HLR.git
cd HLR
python -m venv .venv
```

在 Windows PowerShell 中激活环境

```powershell
.\.venv\Scripts\Activate.ps1
```

在 macOS 或 Linux 中激活环境

```bash
source .venv/bin/activate
```

安装固定版本的依赖并注册 notebook kernel

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name 3544-project --display-name "Python (3544-project)"
```

### 3.2 准备数据集

从 Harvard Dataverse 上的 Duolingo 原始数据集下载 `settles.acl16.learning_traces.13m.csv.gz`，原项目也提供了该数据入口。[数据集](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME) · [Duolingo HLR 原项目](https://github.com/duolingo/halflife-regression)

创建 `data/raw/`，将压缩文件放在以下位置，保持文件名和 gzip 压缩格式

```text
HLR/
└── data/
    └── raw/
        └── settles.acl16.learning_traces.13m.csv.gz
```

原始文件约 361 MiB。Notebook 会按需生成清洗数据、特征缓存、审计摘要、模型产物和图。数据文件、缓存及自动生成的 `reports/` 目录均已从 Git 中排除。

### 3.3 运行完整 notebook

在项目根目录启动 JupyterLab

```bash
jupyter lab
```

打开 `legacy_full_pipeline.ipynb`，选择 **Python (3544-project)**，执行 **Restart Kernel and Run All Cells**。也可以在 VS Code 中选择同一个 Python 环境运行。

根目录的 notebooks 默认使用英文，[中文完整版](notebooks/zh-CN/legacy_full_pipeline.ipynb)和全部中文拆分版集中在 `notebooks/zh-CN/`。每本都提供语言切换链接。从中文子目录启动时，notebook 会自动定位项目根目录，使用相同的数据和自动生成目录。

该 notebook 内嵌完整的数据处理、模型定义、选参、训练和分析代码。将它与按上述路径放置的原始数据放入单独文件夹，并安装所需依赖，也可以独立运行。十二个主模型在 notebook kernel 内训练，数据预处理和三个种子的消融实验支持复用已生成缓存。

### 3.4 运行拆分版 notebooks

**00–10 共十一本文档**按阶段覆盖相同工作流。运行时保留同目录下的 `experiment_pipeline.py`、`pipeline_cells.py` 和 `pipeline_steps/`，使用已注册的 kernel 打开所需阶段并运行全部单元。

每本都支持从全新 kernel 开始，自动生成缺失的上游结果并复用匹配的模型缓存。因此，在刚克隆的仓库中直接运行 07–10，可能触发模型训练。首次按 00–10 顺序运行更便于逐步检查结果。

各阶段的文件与执行说明见 [Notebook 导览](NOTEBOOKS.md#中文)。重复训练可能产生细小数值差异，尤其是排序相关的诊断指标。

两种语言共用训练实现和缓存标识，切换显示语言可以复用已有的拆分版模型结果。中英文完整版本各自内嵌全部实现，均可独立运行。

## 4. 项目结构

```text
HLR/
├── README.md                         # 英文项目介绍
├── README.zh-CN.md                   # 中文项目介绍
├── NOTEBOOKS.md                      # 中英文 notebook 导览
├── requirements.txt                  # Python 依赖
├── legacy_full_pipeline.ipynb        # 独立完整工作流
├── 00_data_preparation.ipynb
├── 01_eda.ipynb
├── 02_hlr_baselines.ipynb
├── 03_hlr_experiments.ipynb
├── 04_random_forest.ipynb
├── 05_lightgbm.ipynb
├── 06_neural_hlr.ipynb
├── 07_model_comparison.ipynb
├── 08_nonlinearity_ablation.ipynb
├── 09_mae_rmse_analysis.ipynb
├── 10_rf_lightgbm_similarity.ipynb
├── experiment_pipeline.py            # 拆分版的执行与结果缓存
├── pipeline_cells.py                 # 加载共用计算代码段
├── pipeline_steps/                   # 数据处理、训练与评估实现
├── notebooks/
│   └── zh-CN/                       # 全部 12 本中文 notebook
├── deliverables/                     # 最终报告与演讲材料
└── paper/                            # 参考论文副本
```

运行后会在本地生成 `data/processed/`、`data/interim/`、`data/artifacts/` 和 `reports/`。仓库保留 notebook 内已执行的输出供读者查看，自动生成文件、中间草稿和 notebook 构建脚本均从版本控制中排除。

根目录包含 12 本英文 notebook，`notebooks/zh-CN/` 中包含对应的 12 本中文版。

## 5. 参考文献与成员

### 主要参考文献

- Settles, B., & Meeder, B. (2016). *A Trainable Spaced Repetition Model for Language Learning*. ACL。[论文](https://research.duolingo.com/papers/settles.acl16.pdf)
- Zaidi, A., Caines, A., Moore, R., Buttery, P., & Rice, A. (2020). *Adaptive Forgetting Curves for Spaced Repetition Language Learning*. AIED。[论文副本](paper/Adaptive%20Forgetting%20Curves%20for%20Spaced.pdf)
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

### 项目成员

ELEC3544，Fall 2026，Group 4

| 成员 | 学号 | 贡献比例 |
| --- | --- | ---: |
| Fu Jing | 3036067121 | 50% |
| Chen Mingjia | 3036068553 | 50% |

两位成员共同参与想法形成、数据清洗、探索性数据分析、建模、训练和结果分析。所有组员的贡献相等，且所有组员都同意上述分配。详细贡献和 AI 工具使用声明见最终报告。
