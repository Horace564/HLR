# Duolingo Half-Life Regression (HLR) 复现指南

> **项目类型**：Data Science / Reproducibility Study  
> **核心论文**：Settles & Meeder (2016), *A Trainable Spaced Repetition Model for Language Learning*, ACL 2016  
> **原则**：先完成可核对的原论文复现，再做新模型、特征和鲁棒性分析。每个 Checkpoint 必须留下可检查的产出。

## 0. 项目简介

### 我们要研究什么？

利用 Duolingo 的学习历史预测学习者在下一次练习中对某个词项的**回忆正确比例**（`p_recall`，范围 0–1）。一条记录对应一个 **user × lexeme × lesson/practice session**；它不是一道题，也不是一个 Token。

原始 HLR 假设记忆按指数曲线遗忘：

\[
\hat h=2^{\boldsymbol\theta^\top\boldsymbol x},\qquad
\hat p=2^{-\Delta/\hat h}
\]

- `x`：当前练习**之前**的历史表现，以及可选的词项特征。
- `Δ`：距离上次练习该词项的时间（模型中按**天**计算）。
- `ĥ`：预测的半衰期（天）；`p̂`：最终预测的回忆概率。
- `y = p_recall`：当前 session 中该词项的实际正确比例。它是观察到的比例，不能直接视作无噪声的真实心理学概率。

**核心研究问题**：只有 3 个可学习参数的 `HLR -lex`，与包含词项特征的 HLR、移除 half-life loss 的变体及其他传统 Baselines 相比，表现怎样？

**本指南的两条实验路线**：

1. **Track A：论文复现（Primary）**——尽量遵循官方数据行顺序、特征、超参数、目标函数和 90/10 切分，结果和 ACL 2016 Table 2 对照。
2. **Track B：规范化 DS 实验（Extension）**——固定 Train / Validation / Test，调参只用 Validation，Test 最终只评估一次；补上 EDA、数据质量分析、误差分组及后续 LightGBM。**两个 Track 的分数不得混在同一张“公平模型排名”表中。**

### 参考与数据链接(需要时再读，如果你找不到相关资料或者很麻烦的话就叫我手动给你提供)

- 原论文：https://aclanthology.org/P16-1174/
- 官方代码：https://github.com/duolingo/halflife-regression
- 官方实现：https://github.com/duolingo/halflife-regression/blob/master/experiment.py
- 官方评估：https://github.com/duolingo/halflife-regression/blob/master/evaluation.r
- **原始公开数据**（Harvard Dataverse，约 361 MB gzip）：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME
- 数据字段说明：https://github.com/duolingo/halflife-regression/blob/master/README.md

> **来源与边界**：官方仓库提供已汇总的学习记录、模型及离线评估代码；**没有**从原始 Exercise 事件到 session 记录的完整上游清洗流水线，也没有完整的 EDA Notebook。以下清洗、EDA、Validation 设计是本项目额外需要完成的工作，而非声称原作者这样做过。

---

## 建议的项目结构

直接一个ipynb过。

---

## Checkpoint 0 — 环境、源码和实验约定

**目标**：确保任何一次模型结果都能追溯到代码、数据、切分、参数。

**任务**

- [ ] 创建 Git 仓库、目录及 Python 3 虚拟环境。
- [ ] 克隆官方代码，阅读 `README.md`、`experiment.py`、`evaluation.r`。
- [ ] 记录 Python 版本、包版本、数据文件名及随机种子（推荐 `42`）。
- [ ] 约定所有模型使用相同样本和同一划分，主要评价指标为 **session-level MAE**。

```bash
mkdir hlr-reproduction && cd hlr-reproduction
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install numpy pandas scipy scikit-learn matplotlib pyarrow jupyter pytest
pip freeze > requirements.txt
mkdir -p data/raw data/interim data/processed notebooks src tests configs results reports/figures external
git clone https://github.com/duolingo/halflife-regression.git external/halflife-regression
```

> **重要兼容性提示**：官方 `experiment.py` 是旧式 Python 代码，含 `iteritems()`、`intern()`、以 `'rb'` 读取 CSV 等写法；**不能保证直接在现代 Python 3 中运行**。本指南推荐用 Python 3 重新实现，同时以官方代码核对细节。如另建旧版 Python/PyPy 环境跑官方脚本，请把它与现代实现隔离。

**产出**：`requirements.txt`、`README.md`、可运行的 Python 环境及固定的实验约定。  
**完成标准**：能够导入 `pandas/numpy/scipy/sklearn`；已打开并理解官方脚本的 `read_data()`、`train_update()`、`predict()`。

---

## Checkpoint 1 — 下载、预览和理解数据（第 1 周）

**目标**：知道每一行代表什么、每一列何时可用，而不是直接把 CSV 扔进模型。

**任务**

- [ ] 从 Harvard Dataverse 下载官方 gzip CSV，原样保存在 `data/raw/`；记下真实文件名、大小、SHA-256。
- [ ] 检查 gzip 文件是否完整、能否解析 CSV header。
- [ ] 使用少量行预览，避免一次性加载约 1,300 万行导致内存不足。
- [ ] 写一份 Data Dictionary（名称、类型、单位、含义、预测时是否可用）。

```bash
# 把 <原始下载文件> 替换成实际文件名；不必解压
# gzip -t data/raw/<原始下载文件>.csv.gz
sha256sum data/raw/*.gz   # macOS 可用 shasum -a 256
```

```python
import glob
import pandas as pd

path = glob.glob('data/raw/*.gz')[0]
df = pd.read_csv(path, compression='gzip', nrows=100_000)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include='all').T)
```

**重点字段**（以官方 README 为准）：

| 字段 | 含义 | 可用于当前 Session 的预测输入？ |
|---|---|---|
| `timestamp` | 当前 lesson/practice 时间戳 | 可用于排序和时间分析 |
| `delta` | 与上一次包含该词项的练习之间的**秒数** | **是**，先换算成天 |
| `history_seen` | 当前 session **之前**遇见该词项的次数 | **是** |
| `history_correct` | 当前 session **之前**正确回忆该词项的次数 | **是** |
| `lexeme_id` / `lexeme_string` | 词项标识及形态标签 | 完整 HLR 可使用 |
| `user_id` | 匿名学习者 ID | **原始 HLR 不使用**，但用于分组分析 |
| `p_recall` | 当前 session 的正确回忆比例 | **Target，不能做输入** |
| `session_seen` | 当前 session 中该词项出现次数 | 不纳入本项目预测输入 |
| `session_correct` | 当前 session 正确次数 | **绝不能做输入**（直接泄漏 Target） |

**产出**：`notebooks/01_data_preview.ipynb`、`reports/data_dictionary.md`、原始文件校验记录。  
**完成标准**：能准确解释 `p_recall` 与 `session_correct / session_seen` 的关系、`delta` 的单位，以及为何 session 列不能直接作为 Predictor。

---

## Checkpoint 2 — 数据质量检查 / Cleaning（第 2 周）

**目标**：构建**可解释、可复现**的清洗规则，并保留与原论文对照的能力。

**检查清单**

- [ ] 各字段的缺失率、数值范围、唯一值数量、异常数据类型。
- [ ] `0 <= p_recall <= 1`；`delta >= 0`；历史计数必须非负。
- [ ] `0 <= history_correct <= history_seen`。
- [ ] `0 <= session_correct <= session_seen`，`session_seen > 0`。
- [ ] 验证 `p_recall ≈ session_correct / session_seen`（允许合理浮点误差）。
- [ ] 检查完全重复行；**不要**仅凭相同的 `(user_id, lexeme_id, timestamp)` 就武断认定是重复记录，先看 session 粒度。
- [ ] 检查 `timestamp` 范围、`delta == 0`、极大 `delta`、不常见语言组合。
- [ ] 保存每条清洗规则影响的行数，不静默删除异常值。

**重要决策**

1. **Track A**：尽量不改变官方输入数据，除非不能解析；异常先记录，不让“自行清洗”悄悄改变复现条件。
2. **Track B**：定义并执行确定性的有效性规则，额外保存排除记录数量与原因。
3. `p_recall=0` 或 `1` **不是脏数据**，不要删除；计算训练用的“观察半衰期”时再按官方方式 clipping。
4. `session_correct/session_seen` 可以用于 **Target 一致性检查**，不能从中构造 Predictor。

**产出**：`reports/data_quality.md`、清洗规则和计数、`src/data.py`。  
**完成标准**：任意一行被保留/删除都有明确理由；清洗前后行数可以对账。

---

## Checkpoint 3 — EDA：理解学习与遗忘数据（第 2–3 周）

**目标**：用数据分布指导特征设计、损失函数与误差分析。

**建议先制作以下 7 张图**

1. `p_recall` 直方图：检查 0、1 处的集中程度和类别不平衡。
2. `delta`（天）的直方图 / log-scale 分布：看长尾。
3. `history_seen` 与 `history_correct` 的分布。
4. 按 `delta` 区间统计平均 `p_recall`（并显示样本数）。
5. 按 `history_seen` 区间统计平均 `p_recall`。
6. 不同 `learning_language` / `ui_language` 组合的样本量和回忆比例。
7. 每位用户、每个词项的观测次数分布（研究数据稀疏性及 cold start）。

数据太大时，先用**来自全文件各部分的抽样**做 EDA，不要只取文件开头 10 万行就认为代表全数据。例如：

```python
import pandas as pd

# 近似随机采样：每个 chunk 抽取 2%，对全文件覆盖更均匀
parts = []
for i, chunk in enumerate(pd.read_csv(path, compression='gzip', chunksize=200_000)):
    parts.append(chunk.sample(frac=0.02, random_state=42 + i))
eda = pd.concat(parts, ignore_index=True)
```

**需要写在 Notebook 里的讨论**

- `delta` 与 `p_recall` 的关系**不一定是随机实验中的遗忘效应**：Duolingo 本来就会根据学习者表现调整复习时间，存在 adaptive scheduling 的选择偏差。
- 一个用户可能跨 Train / Test 出现；这符合“已知用户的未来表现预测”，却不等同于“新用户泛化”。
- 只报告平均值不够，检查极端值和每个分组的样本数量。

**完成标准**：至少记录 **3 条可检验的建模假设**（例如：长间隔下是否误差更高？低历史次数是否难预测？）。

---

## Checkpoint 4 — 固定数据拆分，先防止 Leakage（第 3 周）

**目标**：在训练之前就固定评估协议。**绝对不要反复用 Test Set 选模型。**

### Track A：论文对照协议（不调参）

官方 `experiment.py` 的真实切分代码是：

```python
splitpoint = int(0.9 * len(instances))
train = instances[:splitpoint]
test  = instances[splitpoint:]
```

也就是：**文件顺序前 90% / 后 10%**，然后训练集内部随机 shuffle。

- **不要直接把“文件顺序”写成“时间顺序”**：先检查 `timestamp` 是否严格或基本递增。
- 使用论文/官方代码的默认超参数，先不看 Test 结果做调参。
- 所有 Baselines 使用同一组行索引。
- 记录输入行顺序的哈希/文件版本；设置随机种子以减小实现间随机波动（官方原脚本没有固定种子）。

### Track B：用于开发和调参的推荐协议

固定一个主实验切分：

| 用途 | 全部数据中的比例 | 用法 |
|---|---:|---|
| **Train** | 前 **72%** | 拟合模型参数 `θ` / 树模型 |
| **Validation** | 接下来 **18%** | 选择 `α`、`λ`、学习率、模型参数等 |
| **Test** | 最后 **10%** | 在最终模型锁定之后**仅评估一次** |

这样等价于：**先按原论文保留最后 10% Test，再将前 90% 按 80/20 分成 Train/Val**。

```python
# 注意：这里的 n 是全文件行数，不能用 Checkpoint 1 的 100,000 行预览 df！
n = sum(len(chunk) for chunk in pd.read_csv(path, compression='gzip', chunksize=200_000))
train_end = int(0.72 * n)
test_start = int(0.90 * n)
train_idx = range(0, train_end)
val_idx   = range(train_end, test_start)
test_idx  = range(test_start, n)
```

实际大数据实现可以记录 CSV 行索引、按块处理，而不必把 1,300 万行全部加载到 Pandas。

**标准流程**：

1. 在 Train 上拟合模型。
2. 在 Validation 上选超参数和模型方案；特征工程如果有“学习词表/均值/统计量”的步骤，也只在 Train 拟合。
3. 选好超参数后，用 **Train + Validation = 前 90%** 从头重新训练。
4. 最后只在保留的 **10% Test** 评估一次并冻结结果。

**关于调参数据（与原论文的关系）**：原论文提到使用早期约 100 万条记录来探索训练参数；公开脚本本身没有专门实现超参数搜索和独立 Validation。为了可审计，**我们采用上面的显式 Validation 切分**。调试可以先用 Train 范围内的 10 万～100 万条记录，但这些样本实验的分数不要当作最终测试分数。

**额外实验（可选）**：另做一次明确按 `timestamp` 排序的 **Temporal Holdout**（如 72/18/10），用来检验真正面向未来的预测。它是不同评估协议，**不可与 Track A 的文件顺序分数混比**。
 
**完成标准**：任意给定一行都能确定属于哪个集合；两个集合不共享同一行；所有模型重用完全相同的 split。

---

## Checkpoint 5 — 精确复刻 HLR 特征工程（第 3–4 周）

**目标**：严格按官方 `read_data()` 建立最简特征，再实现完整 Lexeme 特征。

### 5.1 HLR -lex（3 个参数）

\[
\boldsymbol x=\big[\sqrt{1+c},\ \sqrt{1+w},\ 1\big]
\]

其中：

- `c = history_correct`
- `w = history_seen - history_correct`
- 第三维 `1` 是 bias。

**注意**：虽然原论文文本讨论过 `seen/correct/incorrect` 三种计数，**开源实现中实际上只加入了 `sqrt(1+correct)`、`sqrt(1+incorrect)` 和 bias**。不应再把 `history_seen` 作为独立的第 4 维。

```python
import numpy as np

def hlr_lexless_features(df):
    c = df['history_correct'].to_numpy(dtype=np.float64)
    w = (df['history_seen'] - df['history_correct']).to_numpy(dtype=np.float64)
    return np.column_stack([np.sqrt(1.0 + c), np.sqrt(1.0 + w), np.ones(len(df))])
```

### 5.2 完整 HLR

\[
\boldsymbol x_{full} = [\sqrt{1+c},\sqrt{1+w},1,\text{lexeme one-hot}]
\]

- 对 `(learning_language, lexeme_string)` 建稀疏 one-hot；**官方公开脚本**的特征键是 `"learning_language:lexeme_string"`。
- `lexeme_id` 和 `lexeme_string` 是对应的词项标识，但**不要假定直接替换一定严格复现官方脚本**。
- 约 20k 个词项时应使用 `scipy.sparse`、dict 或按活跃特征更新，不要创建 dense 的千万行 × 2 万列矩阵。
- Vocabulary 只从 Train 建立；Val/Test 未出现过的词项按“未知词项、该 one-hot 全 0”处理并记录占比。

### 5.3 时间、Target 与半衰期

```python
delta_days = delta_seconds / 86400.0
p_for_h    = np.clip(p_recall, 0.0001, 0.9999)
h_observed = -delta_days / np.log2(p_for_h)
h_observed = np.clip(h_observed, 15.0 / 1440.0, 274.0)
```

**原实现的数值细节**：官方 `read_data()` 在生成训练样本时也先将观察 `p_recall` clip 到 `[0.0001, 0.9999]`，然后用这个值计算观察半衰期，并以 clip 后的值评估。Track A 对照官方时要保持一致；Track B 可以另报未 clip 的真实比例 MAE，但要标清口径，不要直接混比。

训练用的 `h_observed` 是根据 **Target `p_recall` 推导的辅助监督目标**，不是预测时可取得的输入特征。严格区分：

- `h_observed`：只用于训练的 half-life loss 和离线评估。
- `h_predicted`：模型由历史特征预测出来的半衰期。

**必须写单元测试**

- [ ] `history_seen=0`、`history_correct=0` 时，前两维都是 `1`。
- [ ] `history_seen=10`、`history_correct=8` 时，`x=[sqrt(9),sqrt(3),1]`。
- [ ] 输入矩阵中不含 `p_recall`、`session_correct`、`session_seen`。
- [ ] `delta=86400` 秒正好转换为 1 天。
- [ ] `p=0` 和 `p=1` 不产生 NaN/Inf 的 half-life target。

**产出**：`src/features.py`、`tests/test_features.py`、特征名称列表和维度表。  
**完成标准**：与官方 `read_data()` 对相同样例行生成的特征数值一致。

---

## Checkpoint 6 — 自己拟合 HLR + 理解 Loss（第 4–5 周）

**目标**：先实现最容易核验的 3 参数 `HLR -lex`，再扩展到完整 HLR 和消融实验。

### 6.1 前向预测

\[
\hat h_i=\mathrm{clip}\left(2^{\boldsymbol\theta^\top x_i},\,15/1440,\,274\right)
\]

\[
\hat p_i=\mathrm{clip}\left(2^{-\Delta_i/\hat h_i},\,10^{-4},\,0.9999\right)
\]

### 6.2 Loss

\[
\mathcal L(\theta)=\sum_i\left[(p_i-\hat p_i)^2+\alpha(h_i-\hat h_i)^2\right]+\lambda\lVert\theta\rVert_2^2
\]

- `α`：half-life loss 的权重。
- `λ`：L2 正则化权重。
- **`-h`/`-t` 消融**只是设置 `α=0`，**仍然预测半衰期 `ĥ` 并通过遗忘曲线得到 `p̂`**。
- 默认代码参数：`α=0.01`、`λ=0.1`、学习率 `η=0.001`。代码采用按特征出现次数衰减的学习率，并对 HLR 更新引入 `1/(1+p)` 因子；不是普通的固定学习率 SGD。

### 6.3 建议的实现顺序

**Phase 1：验证数学公式**

- [ ] 用 NumPy 实现 `predict_half_life()`、`predict_recall()`、`loss()`。
- [ ] 用小型合成数据（例如 30–100 行）验证各形状、clip、数值稳定性。
- [ ] 尝试 `scipy.optimize.minimize` 拟合 3 个参数，确认 Loss 能下降。

**Phase 2：复现论文优化细节**

- [ ] 对照官方 `train_update()` 实现在线参数更新。
- [ ] 固定 shuffle seed；保留一轮训练、特征出现次数、学习率衰减及正则项更新细节。
- [ ] 先运行 `HLR -lex`，输出 3 个参数、训练 Loss、Validation/Test MAE。
- [ ] 再加入稀疏 Lexeme one-hot，运行完整 HLR。

> **注意**：用 SciPy 最小化批量 Loss 可以证明模型数学上可拟合，但其优化结果可能不同于原论文的一轮、逐样本、逐活跃特征更新。请把它命名为 `HLR-lex (SciPy)`，不要和“官方优化策略复现”混为同一实现。

**产出**：`src/hlr.py`、`tests/test_hlr.py`、`results/hlr_lexless.weights`、训练曲线/日志。  
**完成标准**：预测落在合法范围、Loss 下降、同一 seed 的结果可复现，参数数量与设计一致。

---

## Checkpoint 7 — 完整 Baselines + Ablation Matrix（第 5–6 周）

**目标**：复现 2016 论文 Section 4.1 的关键比较，而不仅训练一个 HLR。

### 7.1 必做模型清单

| ID | Model | Lexeme 特征 | Half-life loss | 备注 |
|---|---|---|---|---|
| B0 | **Constant Mean** | 否 | 不适用 | 仅用 **Train** 的平均 `p_recall` |
| B1 | **Pimsleur** | 否 | 不适用 | 固定规则，无模型拟合 |
| B2 | **Leitner** | 否 | 不适用 | 固定规则，无模型拟合 |
| B3 | **Logistic Regression** | 是 | 不适用 | 官方 LR 另将 `delta_days` 作为输入 |
| B4 | **LR -lex** | 否 | 不适用 | Logistic Regression 移除词项特征 |
| H1 | **HLR** | 是 | 有 | Full model |
| H2 | **HLR -lex** | 否 | 有 | **3 参数模型** |
| H3 | **HLR -h** | 是 | **无**（`α=0`） | Loss Ablation |
| H4 | **HLR -lex-h** | 否 | **无**（`α=0`） | 双重 Ablation |

可选：`Historical Accuracy` 启发式，例如官方代码中已计算但未作为独立对照输出的 `(history_correct+2)/(history_seen+4)`；记得在报告中标记为**我们新增的 baseline**，不是论文 Table 2 的官方模型。

### 7.2 官方 CLI 命令对照

> **仅用于语义对照**；如下命令要在修复 Python 3 兼容性后运行，或在对应旧版环境中执行。`DATA` 需替换为真实 gzip CSV 路径。

```bash
DATA="data/raw/<your-data-file>.csv.gz"

python external/halflife-regression/experiment.py "$DATA"           # H1: HLR
python external/halflife-regression/experiment.py -l "$DATA"        # H2: HLR -lex
python external/halflife-regression/experiment.py -t "$DATA"        # H3: HLR -h
python external/halflife-regression/experiment.py -l -t "$DATA"     # H4: HLR -lex-h
python external/halflife-regression/experiment.py -m lr "$DATA"     # B3: LR
python external/halflife-regression/experiment.py -m lr -l "$DATA"  # B4: LR -lex
python external/halflife-regression/experiment.py -m leitner "$DATA" # B2
python external/halflife-regression/experiment.py -m pimsleur "$DATA"# B1
```

- 可用 `-x 100000` 做快速 Smoke Test；**`-x` 只读取前 N 行，再把这 N 行按 90/10 切分**，不代表完整数据的最终 Test。
- `B0 Constant` **官方脚本未直接提供模型选项**，需要你们自行实现；用训练集均值，不能用 Test 均值。
- 为保证 Track B 使用统一的 Train/Val/Test，建议由**你们的 `src/train.py` 统一控制切分**，不要让每次调用的官方脚本各自重新划分。

### 7.3 调参方案（只针对 Track B）

从少量组合开始，不必盲目 Grid Search：

| 参数 | 初始候选 | 来源/用途 |
|---|---|---|
| `α`（half-life loss） | `0`, `0.001`, `0.01`, `0.1` | `0` 就是 `-h` 变体 |
| `λ`（L2） | `0`, `0.01`, `0.1`, `1.0` | 控制权重大小 |
| 学习率 `η` | `0.0003`, `0.001`, `0.003` | 训练稳定性 |
| 训练轮数 | `1`, `2`, `5` | 官方默认一轮；多轮属扩展 |

**流程**：Train 拟合 → Validation MAE 选择 → 锁定参数 → Train+Val 重训 → 一次 Test。  
为控制成本，先选固定的 Train 内部子集调试，再在完整 Train/Val 上确认。保存每次试验的 config、seed 和日志。

**产出**：`results/model_comparison.csv`、各模型独立预测文件与权重、消融对照表。  
**完成标准**：H1–H4、B0–B4 全部产生可比较的 **同一 Test** 的预测；能够解释任何一组 ablation 究竟移除了什么。

---

## Checkpoint 8 — 评估、统计分析与误差诊断（第 6–7 周）

**目标**：不仅报告一个 MAE，还理解模型在什么情况下好或不好。

### 8.1 主要指标

1. **MAE（主指标）**：`mean(abs(p_recall - p_hat))`；按 session 记录等权，与原论文比较时优先用它。
2. **ROC-AUC（复现用辅助指标）**：官方 `evaluation.r` 用 `round(p_recall)` 构造二分类标签，再以 `p_hat` 计算 AUC。**不要把 session 比例误称为独立的 0/1 Token 标签。**
3. **Half-life correlation**：只对能预测 `h` 的模型报告 `h_observed` 与 `h_hat` 的 **Spearman** 相关性。
4. **RMSE（可选）**：便于进一步观察较大误差，但它不是论文 Table 2 的主指标。

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy.stats import spearmanr
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
labels_for_auc = np.rint(y_true).astype(int)  # 接近官方 R 的 round(p)
auc = roc_auc_score(labels_for_auc, y_pred) if len(np.unique(labels_for_auc)) == 2 else np.nan
half_life_corr = spearmanr(h_true, h_pred).statistic
```

**一个官方代码细节**：`experiment.py` 内部有个命名为 `spearmanr` 的函数，但实际手写的是**未做秩变换的 Pearson 相关系数形式**；独立复现 Spearman 时，请用 `scipy.stats.spearmanr` 或官方 R `evaluation.r` 的 `cor.test(..., method='s')`，不要照抄该函数名称就认定其计算正确。

### 8.2 论文结果对照

2016 论文 Table 2 报告的参考值（**不是验收时必须精确等于的数值**）：

- HLR：MAE ≈ **0.128**
- HLR -lex：MAE ≈ **0.128**
- HLR -h 与 HLR -lex-h：MAE 约 **0.350**

偏差可能来自运行环境、随机 shuffle、Lexeme 编码、实际数据版本、优化细节或清洗/切分差异。必须在报告中解释，而不是强行调整结果去匹配论文。

### 8.3 必做误差分析

至少画出：

- `delta_days` 分箱下的 MAE（短期 / 中期 / 长期）及每箱样本数。
- `history_seen` 分箱下的 MAE（cold-start / 高频练习）。
- 按 `learning_language` 分组的 MAE。
- `actual p_recall` vs. `predicted p` 的分箱图（Calibration / Bias）。
- HLR 与 HLR -lex **对同一行记录**的绝对误差差值分布。

可选：使用 bootstrap（优先考虑按 user 分组重采样）给 MAE 差异估计置信区间；这样能避免只根据第三位小数就下结论。

**产出**：`notebooks/03_error_analysis.ipynb`、`reports/figures/`、最终模型对比表。  
**完成标准**：能回答“哪个模型更好”“优势有多大”“在哪些用户/时间间隔上效果不同”。

---

## Checkpoint 9 — 复现报告与可选扩展（第 7–8 周）

**目标**：把代码、数据分析、实验结论组织成一份别人能够复现的 Data Science 项目。

**报告建议结构**

1. **Research Question & Motivation**：为什么需要 Recall Prediction，HLR 的假设是什么？
2. **Dataset & Data Quality**：样本单位、字段、清洗规则和 EDA 发现。
3. **Methodology**：两套 split 协议、特征、HLR 公式、Loss、Baselines。
4. **Experimental Setup**：调参范围、seed、硬件、数据划分、评估指标。
5. **Results**：完整模型表、`-lex` / `-h` 消融、论文结果对照。
6. **Error Analysis**：分组 MAE、Calibration、错误案例。
7. **Discussion & Limitations**：数据是否代表因果遗忘？已知用户 vs 新用户；为何 3 参数模型可能表现很好；不具备线上 A/B 实验环境。
8. **Conclusion & Reproducibility**：最重要发现、复现命令、代码及数据来源。

**可选扩展（完成 H1–H4 后再做）**

- [ ] **LightGBM / Random Forest**：以历史计数、`delta_days` 等为输入，比较理论约束模型与灵活树模型。
- [ ] **Feature Ablation**：加历史正确率、`log1p(delta_days)`、交互项，检查增益。
- [ ] **Cold-start**：在新用户、新词项上单独评价（需要另建分组评估协议）。
- [ ] **Temporal Split**：真正按时间留出未来样本，检查结果是否稳健。

**最终交付物**

- [ ] 能一键重新生成核心实验的脚本及依赖说明。
- [ ] `guide.md` + `README.md` + 分阶段 Notebook。
- [ ] 数据检查报告、EDA 图、模型比较表、消融实验表、误差分析。
- [ ] 训练/验证/测试的分割信息及数据来源、原始文件 SHA-256。
- [ ] 至少一份分析性项目报告或 Slides。

---

## 八周建议时间线（两人协作）

| 周 | 主要目标 | Person A | Person B |
|---|---|---|---|
| 1 | CP0–CP1 | 环境、读代码、数据预览 | 数据字典、来源和字段核对 |
| 2 | CP2–CP3 | 质量检查、清洗记录 | EDA 及可视化 |
| 3 | CP3–CP5 | Split / Leakage 检查 | Feature Engineering、单元测试 |
| 4 | CP5–CP6 | 3 参数 HLR -lex | 官方代码与复现细节核对 |
| 5 | CP6–CP7 | 完整 HLR / loss ablation | Constant、LR、Leitner、Pimsleur |
| 6 | CP7–CP8 | 超参数与正式测试 | 评估脚本、指标校验 |
| 7 | CP8–CP9 | 分组误差/可选模型 | 统计分析、图表与写作 |
| 8 | CP9 | 结果复核、最终报告 | 可复现性检查、展示材料 |

## 最小完成版（MVP）与验收门槛

如果时间有限，**优先级按以下顺序**，不要一开始就上 LightGBM：

1. **MVP-1**：下载并理解数据 → 质量检查 → EDA → 固定 Train/Val/Test。
2. **MVP-2**：自己实现 3 参数 **HLR -lex**；检验 loss 和预测数值。
3. **MVP-3**：实现 **HLR -lex-h**，比较移除 half-life loss 的影响。
4. **MVP-4**：加入完整 HLR / HLR -h、Constant、Logistic、Leitner、Pimsleur。
5. **MVP-5**：统一 Test 对比 + 分组误差 + 消融结论。

**验收成功的标准不是“MAE 必须恰好等于 0.128”**，而是：完整记录训练使用了哪些样本、哪些特征、什么 Loss、什么参数，并能解释与原论文的一致和不一致之处。

---

## 复现时最容易踩的 10 个坑

1. `delta` 原始单位是**秒**，HLR 中要除以 `86400` 转成天。
2. `p_recall` 是**当前 Session 的正确比例**，不是一个单独的 0/1 Token 标签。
3. `session_correct` 是当前 Session 的结果，作为 Predictor 会造成严重 **Data Leakage**。
4. **HLR -lex = 3 个参数**；没有额外独立的 `history_seen` 权重。
5. 原始公开实现的 `sqrt` 是 `sqrt(1+count)`，**不是** `sqrt(count)`。
6. `-t`（论文记作 `-h`）表示**移除 half-life loss**，不是删除半衰期预测。
7. 不要把完整 HLR 的巨大 Lexeme One-Hot 做成 dense 矩阵。
8. 官方脚本按**文件行顺序**做 90/10，不能未经验证就宣称“按时间切分”。
9. **Test 不能用来调参**；Track B 使用 Validation，并固定所有模型的数据划分。
10. 官方 `experiment.py` 中叫 `spearmanr` 的函数并不是严格的 Spearman；用 SciPy/R 独立核对。

> **下一步操作**：从 **Checkpoint 0 和 1** 开始。先让 `01_data_preview.ipynb` 成功输出字段、样例、类型及完整性检查，再进入 EDA。此时暂时不训练任何模型。
