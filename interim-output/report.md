# 模型结构、预测目标与特征对间隔重复回忆率预测的影响

## 1. Introduction

间隔重复利用复习之间的时间间隔帮助学习者保持记忆。要描述词元随时间遗忘的过程，需要估计学习者再次遇到该词元时的回忆率。一个有意义的研究问题是：是否可以通过学习者过往的表现来预测下一次学习的表现？在本项目及相关工作中，预测值 `p_recall` 是一次学习或练习 session 中该词元被正确回忆的比例；它是记录级的观测值。

Half-Life Regression（HLR）先根据学习历史估计记忆半衰期，再结合距上次练习的时间计算回忆率。原研究在 Duolingo 约 1290 万条学习记录上评估了这一方法 (Settles & Meeder, 2016)。

Zaidi 等人在 HLR 基础上研究了语言学、心理学和用户特征，并尝试用神经网络表示遗忘曲线。他们从同一 Duolingo 数据中筛选英语学习者，使用约 428 万条记录 (Zaidi et al., 2020)。

受到上述两篇工作的启发，本项目想要研究的问题是：**不同的模型结构、预测目标和新增特征如何影响 `p_recall` 的预测表现？**我们使用清洗后的完整发布数据，统一比较 HLR、用户与语言特征扩展、神经半衰期模型、随机森林和 LightGBM，并分析不同评价指标为何给出不同的模型排序。

## 2. Data and Preprocessing

### 2.1. Dataset

我们使用 Duolingo 发布的学习记录 (Settles & Meeder, 2016)，共 **12,854,226 行、12 列**。每行是一名学习者对一个词元的一次学习或练习 session。各字段含义如下：

| Variable | Description |
|:--|:--|
| `p_recall` | 本次正确回忆比例 |
| `timestamp` | 本次 session 的时间戳 |
| `delta` | 距上次练习的时间（秒） |
| `user_id` | 学习者标识 |
| `learning_language` | 学习语言 |
| `ui_language` | 界面语言 |
| `lexeme_id` | 词元标识 |
| `lexeme_string` | 词元文本标记 |
| `history_seen` | 此前练习次数 |
| `history_correct` | 此前正确次数 |
| `session_seen` | 本次练习次数 |
| `session_correct` | 本次正确次数 |

其中，`p_recall = session_correct / session_seen`，是 session 内的比例。

### 2.2. Data cleaning

我们先按全部 12 列检查完全重复的行，发现 **81** 条，保留首次出现的记录并删除其余重复行，得到 **12,854,145** 行。随后检查所有字段是否缺失，并检查数值字段的类型、有限值、取值范围及计数关系，包括 `p_recall` 是否与 `session_correct / session_seen` 一致。清洗后的数据中没有记录违反这些基础有效性规则。

我们还发现 **101,466 行**数据存在历史计数异常。例如，较晚记录的 `history_seen` 可能小于先前记录的 `history_seen + session_seen`。同一学习者和词元的记录在时间上相互关联，直接删除异常行可能使后续历史计数难以解释；原 HLR 工作未提供可直接采用的修复规则 (Settles & Meeder, 2016)。因此我们保留这些行，不改写计数。

### 2.3. Feature preparation

首先，将 `delta` 从秒换算为天，记为 `Δ = delta / 86400`。其次，将回忆率裁剪为 `p = clip(p_recall, 0.0001, 0.9999)`，作为共同的训练和评价标签。需要半衰期目标的模型再计算 `h = clip(−Δ / log₂(p), 15/1440, 274)` 天；`h` 由标签p_recall反推，并非独立观测值。

输入特征从此前历史表现构建。令 `c = history_correct`、`w = history_seen − history_correct`，基础向量为 `x_base = [√(1+c), √(1+w), 1]`。在此基础上，各模型使用以下输入：

| Input features | Models |
|:--|:--|
| `x_base` | HLR -lex、N-HLR -lex、RF -h、LGBM -h |
| `x_base` + 词元 one-hot | HLR、N-HLR |
| `x_base` + 词元 one-hot + 用户和/或语言 one-hot | HLR-User、HLR-Lang、HLR-User+Lang |
| `x_base` + `Δ` | RF-p、LGBM-p |
| 无输入特征 | Train mean |

词元类别由 `learning_language:lexeme_string` 构成；用户与语言分别由 `user_id`、`learning_language` 和 `ui_language` 编码。为防止标签泄露，本次 session 的 `session_seen` 和 `session_correct` 不作为输入。

### 2.4. Limitations and potential biases

保留历史计数异常行可能使部分历史特征带有噪声。如果异常集中于某些学习者、词元或语言，模型比较也可能受到偏倚。现有数据无法判断这种影响的方向或大小，因此结果应理解为基于保留这些记录的离线数据所得。

## 3. Methodology


### 3.1. Baselines

**Baseline 1 — HLR.** HLR 以记忆半衰期表示遗忘过程 (Settles & Meeder, 2016)。我们使用 `x_base` 加训练集词元 `learning_language:lexeme_string` 的 one-hot 特征作为输入 `x`，先估计半衰期，再由遗忘曲线预测回忆率：

$$
\hat h=\operatorname{clip}(2^{\theta^\top x},15/1440,274),
\qquad
\hat p=\operatorname{clip}(2^{-\Delta/\hat h},0.0001,0.9999).
$$

训练损失同时考虑回忆率、半衰期和参数正则化：

$$
L_{\mathrm{HLR}}=\sum_i\left[(p_i-\hat p_i)^2+0.01(h_i-\hat h_i)^2\right]+0.1\lVert\theta\rVert_2^2.
$$

HLR 用一轮逐样本更新训练，初始学习率为 `0.001`，并按特征出现次数衰减。

**Baseline 2 — HLR -lex.** 只去掉 HLR 的词元 one-hot，保留三个基础输入及相同的预测公式、损失和训练方法，以检查词元特征的作用。

**Baseline 3 — Train mean.** 对所有记录预测训练集 `p` 的均值，不使用输入特征，也不预测半衰期。

### 3.2. Additional features for HLR

以下模型都以完整 HLR 的输入 `x_HLR = [x_base, 词元 one-hot]` 为基础。加入的类别特征用于检验学习者和语言信息是否能解释仅靠练习历史与词元无法捕捉的差异。

**HLR-User.** 输入为 `[x_HLR, user_id one-hot]`。Train 中共有 **92,392** 名用户，因此新增 92,392 个特征，总输入维度为 **110,748**。我们加入用户标识，是为了让模型学习不同学习者的系统性回忆差异。

**HLR-Lang.** 输入为 `[x_HLR, learning_language one-hot, ui_language one-hot]`。两列语言分别增加 **6** 维和 **4** 维，总输入维度为 **18,366**。我们加入这两列，是为了检验学习语言和界面语言是否对应不同的回忆表现。

**HLR-User+Lang.** 输入同时包含 `x_HLR`、用户 one-hot 和两列语言 one-hot，总维度为 **110,758**。这一版本检验语言信息在已有用户特征的基础上是否还能提供帮助。

三个模型均使用第 3.1 节的遗忘曲线、损失函数和训练方法。

### 3.3. Neural half-life regression

参考神经半衰期模型 (Zaidi et al., 2020)，我们保留 HLR 的遗忘曲线，把线性的 `θᵀx` 换成含 **4 个 ReLU 隐藏单元**的网络，以检验非线性特征组合是否有帮助。网络输出为：

$$
\log_2\hat h=W_2^\top\operatorname{ReLU}(W_1^\top x)+b.
$$

再按第 3.1 节的公式计算 `ĥ` 和 `p̂`。

**N-HLR** 使用基础输入与词元特征，**N-HLR -lex** 仅使用基础输入。两者由对应的线性 HLR 权重初始化，使用损失 `MSE(p) + 0.0001 MSE(log₂ h) + 10⁻⁶ L2`。

| 模型 | 超参数 |
|:--|:--|
| N-HLR | 隐藏单元 4；Adam；学习率 0.0001；批量 8192；最多 5 轮；Validation 选定第 5 轮 |
| N-HLR -lex | 隐藏单元 4；Adam；学习率 0.0001；批量 8192；最多 5 轮；Validation 选定第 5 轮 |


### 3.4. Tree-based models

#### 3.4.1. 模型介绍

除了神经网络，我们还检验基于决策树的方法能否学习更复杂的非线性关系。我们采用 Breiman (2001) 提出的随机森林（RF）和 Ke et al. (2017) 提出的 LightGBM，并为每种算法设置两种训练目标：先预测半衰期、再代入遗忘曲线计算 `p̂`，或直接预测 `p`。组合后共有四个模型，均使用平方误差训练，不加入词元、用户或语言 one-hot：

- **RF -h：** RF 使用 `x_base` 预测 `log₂(h)`，转换为半衰期后按 HLR 曲线计算 `p̂`。
- **RF-p：** RF 使用 `[x_base, Δ]` 直接预测 `p`，再将预测值裁剪至 `[0.0001, 0.9999]`。
- **LGBM -h：** LightGBM 使用与 RF -h 相同的输入和目标，再按同一遗忘曲线计算 `p̂`。
- **LGBM-p：** LightGBM 使用与 RF-p 相同的输入和目标，直接得到裁剪后的 `p̂`。

Direct-p 模型不直接预测半衰期；从 `p̂` 反推的半衰期只用于诊断。由于 direct-p 版本还增加了 `Δ` 输入，它与半衰期版本的差异不能单独归因于预测目标。

| 模型 | 超参数 |
|:--|:--|
| RF -h | `n_estimators=100`, `max_depth=8`, `min_samples_leaf=50` |
| RF-p | `n_estimators=100`, `max_depth=12`, `min_samples_leaf=50` |
| LGBM -h | `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=50`, 最佳轮次 271 |
| LGBM-p | `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=200`, 最佳轮次 333 |

#### 3.4.2. 选参策略

RF 的六组候选为树深 `8/12/16` × 最小叶节点样本数 `50/100`，每组使用 **100** 棵树；LightGBM 的六组候选为叶子数 `15/31/63` × 最小叶节点样本数 `50/200`，学习率为 `0.05`。四个模型共用固定抽取的 **20 万条 Train** 进行候选拟合，分别按完整 Validation 的 `MAE(p)` 选配置，再用全部 Train 重新拟合。LightGBM 最多训练 **1000** 轮；候选及全量拟合都以 Validation 目标均方误差进行 **50** 轮提前停止。



## 4. Experiments

### 4.1. Data split and model selection

清洗后的 **12,854,145** 行按原文件顺序划分为 **Train 72%（9,254,984 行）**、**Validation 18%（2,313,746 行）**和 **Test 10%（1,285,415 行）**。Train 用于模型训练，Validation 用于选参和提前停止，Test 用于最终评估。

### 4.2. Experimental questions and evaluation metrics

实验围绕三个问题：

1. **新增特征是否改善 HLR？** 比较 HLR 与 HLR -lex，以及 HLR 与三个用户/语言特征扩展模型。
2. **非线性模型是否改善预测？** 将 N-HLR 与对应的 HLR 基线比较，并将树模型与 HLR、Train mean 比较。
3. **树模型的预测方式有何影响？** 分别比较 RF -h 与 RF-p、LGBM -h 与 LGBM-p。

十二个模型在同一批 Test 记录上评价。主指标为平均绝对误差（MAE）、均方根误差（RMSE）和 AUC。设 `p_i` 为第 `i` 条记录裁剪后的真实回忆率，`p̂_i` 为预测值，Test 样本数为 `N`，则

$$
\operatorname{MAE}(p)=\frac{1}{N}\sum_{i=1}^{N}|p_i-\hat p_i|,
\qquad
\operatorname{RMSE}(p)=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(p_i-\hat p_i)^2}.
$$

MAE 描述平均误差，RMSE 对大误差更敏感。计算 AUC 时，将 `y_i = np.rint(p_i)` 作为记录级二值标签。设正、负标签的记录数分别为 `N₊`、`N₋`，则

$$
\operatorname{AUC}=\frac{1}{N_+N_-}\sum_{i:y_i=1}\sum_{j:y_j=0}
\left(\mathbf{1}[\hat p_i>\hat p_j]+\tfrac12\mathbf{1}[\hat p_i=\hat p_j]\right).
$$

AUC 衡量正标签记录排在负标签记录前面的能力，并列预测计半分。MAE、RMSE 越低越好，AUC 越高越好。

### 4.3. leakage controls

为避免标签泄漏，本次 session 的 `session_seen/session_correct`、`p` 和由它反推的 `h` 都不作为输入；词元、用户和语言词表也不从 Validation/Test 学习。Test 标签没有用于拟合、提前停止或超参数选择。时间戳核验显示，Train 中最晚记录为 **2013-03-09 03:24:09 UTC**，早于 Test 中最早记录 **2013-03-11 13:45:39 UTC**（相差 210,090 秒），不存在训练记录晚于测试记录的时间倒置。

### 4.4. Computing Environment and Reproducibility

| Item | Setting |
|:--|:--|
| OS | Windows 10 Home，64 位 |
| CPU | Intel Core i5-1135G7，4 核 / 8 线程 |
| Memory | 约 8 GB |
| Compute | CPU；训练未使用 GPU |
| Python environment |Python 3.11.17 |
| Main packages | NumPy 2.4.6；pandas 3.1.1；scikit-learn 1.9.1；LightGBM 4.7.0 |

RF 使用 scikit-learn 实现 (Pedregosa et al., 2011)。实验以随机种子 **42** 为基础；树模型候选抽样使用 42/43，神经网络的训练顺序使用 52。没有使用外部预训练检查点；N-HLR 由本实验的 HLR 权重初始化。主实验每个配置运行一次；第 5.2 节的非线性消融另使用三个配对种子 42/43/44。

源代码可在 `<this github repo>` 找到。

## 5. Results and Discussion

### 5.1. Overall results

表 1–4 比较清洗后同一批 **1,285,415 条 Test 记录**上的十二个模型。MAE 和 RMSE 是回忆率的绝对误差与均方根误差，单位均为回忆率比例（无量纲），越低越好；AUC 衡量模型区分二值回忆结果的排序能力，也无量纲，越高越好。二值标签由 `np.rint(p_target)` 得到，指标定义见 4.2。所有模型使用相同的数据切分和 Test 评估口径。加粗仅标出十二个模型中各列的最佳数值，不表示统计显著性。

**表 1. Baseline 的 Test 结果。**

| 模型 | MAE ↓ | RMSE ↓ | AUC ↑ |
|:--|--:|--:|--:|
| HLR | 0.128033 | 0.290877 | 0.537856 |
| HLR -lex | 0.128283 | 0.290873 | 0.537459 |
| Train mean | 0.174210 | 0.271138 | 0.500000 |

**表 2. Adding features to HLR 的 Test 结果。**

| 模型 | MAE ↓ | RMSE ↓ | AUC ↑ |
|:--|--:|--:|--:|
| HLR-User | 0.127069 | 0.290409 | 0.540654 |
| HLR-Lang | 0.127938 | 0.290823 | 0.537993 |
| HLR-User+Lang | 0.126819 | 0.290242 | 0.540222 |

**表 3. NN 的 Test 结果。**

| 模型 | MAE ↓ | RMSE ↓ | AUC ↑ |
|:--|--:|--:|--:|
| N-HLR | **0.118898** | 0.287320 | 0.539874 |
| N-HLR -lex | 0.118909 | 0.287335 | 0.540314 |

**表 4. Tree-based models 的 Test 结果。**

| 模型 | MAE ↓ | RMSE ↓ | AUC ↑ |
|:--|--:|--:|--:|
| RF -h | 0.164276 | 0.317893 | 0.549080 |
| RF-p | 0.170206 | **0.267520** | 0.604966 |
| LGBM -h | 0.164277 | 0.317891 | 0.549281 |
| LGBM-p | 0.170254 | 0.267539 | **0.605412** |

通过上表的结果，我们有以下发现：

1. 在原始的HLR上新增user_id和学习语言作为额外特征并没有带来太大的提升。HLR-User+Lang 将 MAE 降至 0.126819，比原始HLR有略微提升；单独加入用户特征或单独加入语言特征并没有给结果带来太大的变化。

2. 基于神经网络的N-HLR 和 N-HLR -lex在预测误差MAE上表现最好。与 HLR 相比，**N-HLR 的 MAE 从 0.128033 降至 0.118898**，是全部模型中最低的；加入词元后，N-HLR 的 MAE 几乎不变。

3. 同一预测目标下（即预测h或直接预测p），RF 与 LGBM 在所有指标上的结果几乎一致。

4. 相比之下，对两种树模型来说， 预测目标从`-h` 改为 `-p` 时，两种模型的表现差异较大。直接预测h的模型，Mae表现较好，直接预测p的模型在rmse和AUC上表现较好。比如，随机森林-p的RMSE 从 **0.317893** 降至 **0.267520**，AUC 从 **0.549080** 升至 **0.604966**，但 MAE 从 **0.164276** 升至 **0.170206**；LGBM 也呈现同样趋势。由于 `-p` 版本同时改变了预测目标和输入特征，这些差异不能只归因于预测目标。

5. 不同模型在不同指标上的表现不一。与 HLR 的 **MAE 0.128033、RMSE 0.290877、AUC 0.537856** 相比，RF-p 和 LGBM-p 的 MAE 更高（分别为 **0.170206** 和 **0.170254**），RMSE 更低（**0.267520** 和 **0.267539**），AUC 更高（**0.604966** 和 **0.605412**）。若重视平均绝对误差，N-HLR 更合适；若重视较大误差或已发生记录的二值排序，直接预测回忆率的树模型更有优势。

### 5.2. Discussion

**N-HLR 的改善来自非线性吗？** 原始 HLR 与 N-HLR 同时改变了模型结构、损失和优化方式，因而不能直接把二者差异归因于非线性。为此，我们在相同的 `x_base` 输入上增加 Aligned-Linear-HLR -lex：它保持线性输出，但采用 N-HLR 的损失、Adam、批量训练和 Validation 选模。对齐模型使用三个配对种子 42/43/44，结果见表 5。

**表 5. 非线性消融的 Test MAE。** `±` 后为三个种子间标准差；原始 HLR -lex 为确定性的单次基线。

| 模型 | `log2(ĥ)` 结构 | 训练方式 | Test MAE |
|:--|:--|:--|--:|
| Original HLR -lex | 线性 | 原始 HLR | 0.128283 |
| Aligned-Linear-HLR -lex | 线性 | N-HLR 对齐 | 0.119907 ± 0.000016 |
| N-HLR -lex | 4-ReLU | N-HLR 对齐 | 0.118903 ± 0.000053 |

保持线性结构、只改变训练规则后，MAE 已降低 **0.008376**；在此基础上加入非线性又降低 **0.001004**，且三个种子方向一致。按差值作描述性分解，约 89% 的总改善出现在训练规则对齐后，约 11% 是非线性的额外收益。因此，非线性确有小而稳定的贡献，但并不是 N-HLR 改善的主要来源，主要的改善来源来自Loss函数定义和优化方式。两种参数化的 L2 约束不能严格等价，该比例不应解释为严格的因果贡献率。

**为什么 MAE 与 RMSE 排序相反？** 这是模型比较中的一个实质性结果，而不是两个指标互相矛盾。Test 标签高度不均衡：**83.603%** 等于 `MAX_P`，而 **7.023%** 等于 `MIN_P`。HLR 的预测也高度集中在高回忆率端，**82.96%** 的预测不低于 0.95；RF-p 的这一比例只有 **2.12%**，其输出更集中在约 0.9 附近。于是 HLR 更贴近数量占绝对多数的上界标签，但在低回忆率记录上更容易发生严重高估。

为量化各组对总体指标的作用，我们计算 `C_MAE=(N_g/N)×MAE_g` 和 `C_MSE=(N_g/N)×MSE_g`。前者按组相加得到总体 MAE，后者相加得到总体 MSE，再开平方得到 RMSE。两个决定性边界组见表 6。

**表 6. 上、下边界组对总体误差的贡献。**

| 真实标签组 | Test 占比 | HLR `C_MAE` | RF-p `C_MAE` | HLR `C_MSE` | RF-p `C_MSE` |
|:--|--:|--:|--:|--:|--:|
| `p_target=MAX_P` | 83.603% | 0.029656 | 0.085221 | 0.007673 | 0.009936 |
| `p_target=MIN_P` | 7.023% | 0.066838 | 0.061261 | 0.064389 | 0.053875 |

在上界组，HLR 的 MAE contribution 比 RF-p 低 **0.055565**，这一高频优势使 HLR 的总体 MAE 更低。相反，下界组单独贡献了 HLR 总 MSE 的大部分；平方惩罚放大了这些低频但幅度很大的误差，使 HLR 的 MSE 为 **0.084610**、RMSE 为 **0.290877**，高于 RF-p 的 **0.071567** 和 **0.267520**。因此，MAE 更反映大量高回忆率记录上的典型绝对误差，RMSE 对低回忆率记录上的大误差更敏感。图 1 给出全部五组的加性分解；RMSE 本身不能按组直接相加。

![HLR 与 RF-p 的分组 MAE/MSE contribution](reports/s09_mae_mse_contribution_stacks.png)

**图 1.** 真实回忆率各组对总体 MAE（左）和 MSE（右）的加性贡献。柱尾给出总体 MAE，或总体 MSE 及其平方根 RMSE。

**RF 与 LightGBM 为什么如此接近？** 看到同一目标下两类算法几乎相同的指标时，我们首先怀疑是否存在模型复用、数据索引错位、缓存覆盖或后处理错误。为此，我们分别调用四个已训练模型，在同一 Test 特征上重新预测，并同时比较原始输出和最终输出。重建已保存预测的最大误差低于 `8×10⁻¹⁶`；E2/E4 没有完全相同的记录，E1/E3 的原始输出也不完全相同，其 **23.349%** 最终相等记录可由共同裁剪到边界解释。这些检查排除了所检验的模型混用、索引、缓存和后处理问题。

排除上述问题后，更可能的解释是：在相同数据、低维特征和共同目标下，RF 与 LightGBM 学到了高度相似的主要预测模式。它们仍非逐行等价：E1/E3 和 E2/E4 的平均绝对差异分别为 **0.000678** 和 **0.003580**。本项目只做了初步诊断，没有系统研究这种收敛现象；后续可通过改变训练子样本、随机种子、特征集合和模型容量，并比较残差与特征贡献，检验相似性是否仍然存在。详细结果见第 7 节。




## 6. Conclusion

在相同的清洗数据和 Test 记录上，**N-HLR 的 MAE 最低（0.118898）**，用户及语言特征只小幅改善 HLR。在仅使用 `x_base` 的三种子对齐消融中，N-HLR 的大部分 MAE 增益来自损失和训练规则，四单元 ReLU 非线性另带来约 **0.001** 的稳定改善。RF-p 的 RMSE 最低（0.267520），LGBM-p 的 AUC 最高（0.605412），但两者的 MAE 高于 HLR；高回忆率标签占多数，而低回忆率记录产生少量大误差，解释了 MAE 与 RMSE 的相反排序。逐行核验还表明，同一目标下 RF 与 LightGBM 学到高度相似但并不相同的预测；E1/E3 的边界裁剪进一步放大了这种相似性。因此，没有一个模型在全部指标上最优。

这些结论仅适用于当前按记录切分的离线测试。部分历史计数异常被保留，同一学习者也可能同时出现在训练和测试中；非线性消融的两种参数化也不具有完全等价的 L2 约束，direct-p 树模型则同时改变了目标和输入。未来应扩大随机种子数量，在相同输入下单独比较预测目标，采用按学习者隔离的评估，并检查历史计数异常的影响。RF 与 LightGBM 的相似性目前也只是初步观察；后续可改变训练子样本、特征和模型容量，并比较残差与特征贡献，以检验两类算法是否稳定地学习到相似模式。

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. In K. Erk & N. A. Smith (Eds.), *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1848–1858). Association for Computational Linguistics.

Zaidi, A., Caines, A., Moore, R., Buttery, P., & Rice, A. (2020). Adaptive forgetting curves for spaced repetition language learning. In I. I. Bittencourt, M. Cukurova, K. Muldner, R. Luckin, & E. Millán (Eds.), *Artificial intelligence in education* (Vol. 12164, pp. 358–363). Springer.

## 7. Supplementary Materials: RF–LightGBM paired diagnostics

### 7.1. Aim and design

RF 与 LightGBM 在相同目标下的总体指标几乎一致。由于两种算法的训练机制不同，我们首先怀疑是否存在模型复用、错误特征、Test 行错位、缓存覆盖或相同后处理掩盖差异。为验证这些可能性，我们在全部 **1,285,415 条 Test 记录**上进行逐行配对比较：E1（RF-logh）对 E3（LGBM-logh），以及 E2（RF-p）对 E4（LGBM-p）。每组同时比较最终裁剪后的 `p_hat` 与模型原始输出；E1/E3 的原始空间为 `log2(h)`，E2/E4 为 `p`。

对每组计算 Pearson、Spearman、平均绝对差异及差异分位数，并统计完全相同、共同位于裁剪边界以及仅在内部记录上接近的比例。最后重新调用四个已训练模型，按主流程重建 Test 预测，以检查模型、特征、迭代轮次、缓存索引和后处理是否一致。

### 7.2. Paired prediction results

**表 S1. RF 与 LightGBM 的逐行预测差异。** 原始 `log2(h)` 与概率尺度单位不同，绝对差异不能跨尺度直接比较。

| 配对 | 输出空间 | Pearson | Spearman | 平均绝对差异 | P50 | P90 | P99 | 完全相同 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| E1 vs E3 | clipped `p` | 0.999839 | 0.999821 | 0.000678 | 0.000053 | 0.001461 | 0.009423 | 23.349% |
| E1 vs E3 | raw `log2(h)` | 0.998481 | 0.997860 | 0.015825 | 0.008193 | 0.027927 | 0.158902 | 0% |
| E2 vs E4 | clipped `p` | 0.976632 | 0.984299 | 0.003580 | 0.001861 | 0.006696 | 0.032542 | 0% |
| E2 vs E4 | raw `p` | 0.976632 | 0.984299 | 0.003580 | 0.001861 | 0.006696 | 0.032545 | 0% |

**表 S2. 边界与重建检查。** “内部接近”指两个最终预测均不在裁剪边界且绝对差异小于 `10⁻⁶`。

| 配对 | 同一裁剪边界 | 两者均在内部 | 内部接近 | 最大重建误差 |
|:--|--:|--:|--:|--:|
| E1 vs E3 | 23.349% | 983,133 | 7.401% | <8×10⁻¹⁶ |
| E2 vs E4 | 0% | 1,285,378 | 0.088% | <8×10⁻¹⁶ |

![RF 与 LightGBM 的逐行预测诊断](reports/s10_rf_lgbm_similarity.png)

**图 S1.** 左列为固定抽取 20,000 条 Test 记录的配对预测，虚线为 `y=x`；右列为全部 Test 记录的绝对差异分布，横轴为对数刻度。

### 7.3. Interpretation

E1/E3 的最终 `p_hat` 几乎重合，但 **23.349%** 的完全相同预测恰好等于共同落在相同裁剪边界的比例；原始 `log2(h)` 没有完全相同的记录。因此，两者学习到高度相关但不同的半衰期函数，遗忘曲线变换和裁剪进一步放大了表面相似性。

E2/E4 没有共同边界或完全相同的记录，且原始与裁剪后的差异统计几乎一致，说明这组相似性来自实际模型输出。四个模型都能独立重建对应的已保存预测，误差仅为浮点舍入量级；结合原始输出并不相同，这些检查排除了本实验所检验的模型复用、Test 行错位、缓存覆盖和后处理错误。P99 差异分别为 **0.009423** 和 **0.032542**，所以两类算法仍不能视为逐行等价。

排除上述实现问题后，一个合理假说是：相同数据、低维输入、共同目标和超过九百万条训练记录，使两种树集成方法从相同特征中学到了高度相似的主要预测模式；对 E1/E3，裁剪又进一步缩小了最终差异。但本项目没有系统分析这一现象，也没有证明两种算法为何会收敛到相似映射。后续应在不同随机种子、训练子样本、特征组合和模型容量下重复比较，并分析逐组残差、树结构或特征贡献，作为独立的 future work。
