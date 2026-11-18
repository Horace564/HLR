# 间隔重复回忆率预测：十二模型对照实验

本项目使用 Duolingo 公开的学习记录，研究模型结构、预测目标和新增特征如何影响回忆率 `p_recall` 的预测。所有模型使用同一份去重数据和固定的 Train / Validation / Test 切分；主要比较指标为回忆率的 MAE，并分析 RMSE、AUC、半衰期 Spearman 相关系数及不同复习间隔下的表现。研究背景、方法与结果见 [完整报告](report.md)。

## 快速开始

### 1. 准备 Python 环境

建议使用 64 位 Python 3.11。从仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name 3544-project --display-name "Python (3544-project)"
```

macOS / Linux 将激活命令换成 `source .venv/bin/activate`。项目已在 Python 3.11 的 `data-science` 环境中检查过；`requirements.txt` 列出运行所需的包。已有满足依赖的环境也可以直接使用。

### 2. 获取原始数据

从 Duolingo [HLR 原项目](https://github.com/duolingo/halflife-regression)链接的 [Harvard Dataverse 数据集](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME)下载压缩 CSV，放到以下位置，保持文件名不变，也不要解压：

```text
data/raw/settles.acl16.learning_traces.13m.csv.gz
```

新检出的仓库若没有 `data/raw/`，先在仓库根目录执行 `mkdir data/raw`。

代码会从这个文件自动构建去重数据和特征缓存。原始文件约 361 MiB；完整实验还会产生较大的中间文件和模型文件，请预留数 GB 磁盘空间。数据来自 Settles 与 Meeder（2016）；使用或引用数据时请引用原论文，见下方参考资料。

### 3. 启动并运行 notebook

在仓库根目录运行：

```powershell
jupyter lab
```

在 JupyterLab 中选择 **Python (3544-project)** kernel，打开需要的 notebook，执行 **Restart Kernel and Run All Cells**。也可以用 VS Code 打开 notebook，并选择同一个 Python 环境。

推荐按下表顺序首次运行，便于逐步检查结果；每本 notebook 也能从全新 kernel 单独运行，不需要先执行其他 notebook。若直接运行比较或误差分析，程序会自动补齐缺失的训练结果，首次运行可能耗时较长。

| 顺序 | Notebook | 任务 |
| --- | --- | --- |
| 00 | [数据准备与审计](00_data_preparation.ipynb) | 去重、有效性与历史计数审计、固定切分和特征缓存 |
| 01 | [探索性分析](01_eda.ipynb) | 去重数据的分布与图表 |
| 02 | [HLR 基线](02_hlr_baselines.ipynb) | HLR、HLR -lex、Train mean |
| 03 | [HLR 特征实验](03_hlr_experiments.ipynb) | HLR-User、HLR-Lang、HLR-User+Lang |
| 04 | [随机森林](04_random_forest.ipynb) | RF-logh、RF-p |
| 05 | [LightGBM](05_lightgbm.ipynb) | LGBM-logh、LGBM-p |
| 06 | [神经半衰期模型](06_neural_hlr.ipynb) | N-HLR、N-HLR -lex |
| 07 | [统一模型比较](07_model_comparison.ipynb) | 十二模型的 Validation / Test 与复习间隔对照 |
| 08 | [误差分析](08_error_analysis.ipynb) | 误差分解、模型预测差异与风险排序 |

## 数据与评估口径

原始数据约 1,285 万行、12 列。项目按全部字段去除 81 条完全重复记录，得到 **12,854,145** 行；基础有效性审计通过，历史计数异常单独报告并保留。清洗后的行顺序决定固定切分：Train **9,254,984** 行、Validation **2,313,746** 行、Test **1,285,415** 行，比例约为 72% / 18% / 10%。选参与提前停止使用 Train / Validation；Test 用于最终比较。详细定义见 [报告](report.md)。

## 输出、缓存与重复运行

| 路径 | 内容 |
| --- | --- |
| `data/processed/` | 去重后的 gzip CSV 和清单 |
| `data/interim/` | 特征、类别索引缓存和审计明细 |
| `data/artifacts/` | 模型权重、选参表、训练资源统计、评估结果和逐行 Test 预测 |
| `reports/` | 数据质量与历史计数审计摘要 |

模型结果按源数据、切分、相关计算代码和主要软件版本生成缓存标识。重复运行同一 notebook 时会读取匹配的结果；缺失或不匹配时才重新计算。`07_model_comparison.ipynb` 保存所有模型的 Test 逐行预测，后续分析可复用。随机森林的全量模型文件可能较大，训练阶段也需要较多内存。`joblib` 文件应只从自己信任的本项目运行结果加载。

## 代码结构

- [experiment_pipeline.py](experiment_pipeline.py)：管理 notebook 间的依赖、训练入口及结果缓存。
- [pipeline_steps/](pipeline_steps/)：按阶段保存从原完整 notebook 拆出的可阅读计算代码。
- [pipeline_cells.py](pipeline_cells.py)：加载这些计算段供共享入口使用。
- [legacy_full_pipeline.ipynb](legacy_full_pipeline.ipynb)：拆分前的完整 notebook 归档。
- [build_split_notebooks.py](build_split_notebooks.py)：从归档重新生成拆分后的 notebook 和 `pipeline_steps/`；日常运行实验无需执行。生成时会覆盖这些生成文件。

## 参考资料

- Settles, B. & Meeder, B. (2016). *A Trainable Spaced Repetition Model for Language Learning*. [论文](https://research.duolingo.com/papers/settles.acl16.pdf) · [原项目与数据说明](https://github.com/duolingo/halflife-regression)。
- 项目使用的两篇参考论文副本位于 [paper/](paper/)；项目分析见 [report.md](report.md)。
