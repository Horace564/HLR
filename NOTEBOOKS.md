# Notebook Guide / Notebook 运行导览

[English README](README.md) · [中文 README](README.zh-CN.md)

## English

### Choose a workflow

| Entry point | Required project files | Execution |
| --- | --- | --- |
| `legacy_full_pipeline.ipynb` | The notebook, raw dataset, and installed Python dependencies | Contains all implementations and trains the twelve main models in one kernel |
| Modular notebooks 00–10 | The notebooks, `experiment_pipeline.py`, `pipeline_cells.py`, `pipeline_steps/`, raw dataset, and dependencies | Each notebook starts independently and generates or loads its upstream results |

Use the environment setup and dataset path in [Quick start](README.md#3-quick-start). For a first run, follow the modular notebooks in numerical order, or run the standalone notebook from top to bottom. Select **Python (3544-project)** and use **Restart Kernel and Run All Cells**. The two workflows cover the same models and analyses.

The root directory provides the twelve English notebooks. All twelve Chinese editions are grouped in [`notebooks/zh-CN/`](notebooks/zh-CN/), including the [Chinese standalone notebook](notebooks/zh-CN/legacy_full_pipeline.ipynb). Language links appear at the top of each notebook. Chinese notebooks locate the project root automatically, including when their kernel starts in the Chinese subdirectory.

### Modular stages

| Stage | Notebook | Contents |
| --- | --- | --- |
| 00 | [Data preparation](00_data_preparation.ipynb) | Exact-duplicate removal, validity and historical-count audits, fixed splits, feature caches |
| 01 | [Exploratory data analysis](01_eda.ipynb) | Recall labels, review intervals, practice history, language pairs, learner and lexeme frequencies |
| 02 | [HLR baselines](02_hlr_baselines.ipynb) | HLR, HLR -lex, and the Train mean baseline |
| 03 | [HLR feature extensions](03_hlr_experiments.ipynb) | HLR-User, HLR-Lang, HLR-User+Lang |
| 04 | [Random forest](04_random_forest.ipynb) | RF-logh (E1), RF-p (E2), hyperparameter selection and full-Train fits |
| 05 | [LightGBM](05_lightgbm.ipynb) | LGBM-logh (E3), LGBM-p (E4), selection and early stopping |
| 06 | [Neural HLR](06_neural_hlr.ipynb) | N-HLR and N-HLR -lex, HLR initialization and Validation trajectories |
| 07 | [Model comparison](07_model_comparison.ipynb) | Twelve-model Validation/Test tables, Test predictions, and review-interval diagnostics |
| 08 | [Nonlinearity ablation](08_nonlinearity_ablation.ipynb) | Original HLR -lex, aligned linear HLR, and four-ReLU N-HLR with three paired seeds |
| 09 | [MAE and RMSE analysis](09_mae_rmse_analysis.ipynb) | Label/prediction distributions, high-prediction shares, grouped HLR versus RF-p error contributions |
| 10 | [RF and LightGBM similarity](10_rf_lightgbm_similarity.ipynb) | Paired raw and final predictions, clipping boundaries, and prediction reconstruction checks |

### Execution and generated files

- Run from the folder containing the notebooks. Put the raw gzip at `data/raw/settles.acl16.learning_traces.13m.csv.gz`.
- Keep the dataset under the project root for both languages. Shared training code selects English by default and Chinese through `Pipeline(language="zh-CN")`. Localization does not change model-cache identities.
- Modular notebooks automatically create missing dependencies. A direct run of 07–10 can trigger training of upstream models. Matching main-model artifacts are reused from `data/artifacts/`.
- The standalone notebook fits the twelve main models directly. Data preprocessing and the six matched-training ablation runs can reuse their local caches. Without an ablation cache, 08 trains both aligned architectures for seeds 42, 43, and 44.
- Generated files appear under `data/processed/`, `data/interim/`, `data/artifacts/`, and `reports/`. The workflows create these directories as needed, and Git ignores their contents. Saved notebook tables and plots remain available for inspection.
- Full-data fits take longer and use more memory than reading existing results. Runtime, memory measurements, and some floating-point or rank-based statistics can vary between runs.

## 中文

### 选择运行方式

| 入口 | 所需项目文件 | 执行方式 |
| --- | --- | --- |
| `legacy_full_pipeline.ipynb` | 该 notebook、原始数据和已安装的 Python 依赖 | 内嵌全部实现，在同一个 kernel 中训练十二个主模型 |
| 拆分版 00–10 | Notebooks、`experiment_pipeline.py`、`pipeline_cells.py`、`pipeline_steps/`、原始数据和依赖 | 每本可以独立开始，自动生成或读取上游结果 |

环境准备和数据路径见 [快速开始](README.zh-CN.md#3-快速开始)。首次运行可以按数字顺序执行拆分版，也可以从头到尾执行完整 notebook。选择 **Python (3544-project)**，运行 **Restart Kernel and Run All Cells**。两种方式覆盖相同的模型与分析。

根目录提供十二本英文版，十二本中文版集中在 [`notebooks/zh-CN/`](notebooks/zh-CN/)，其中包含[中文完整 notebook](notebooks/zh-CN/legacy_full_pipeline.ipynb)。每本顶部都提供语言切换链接。中文版本会自动定位项目根目录，可以直接从中文子目录启动 kernel。

### 拆分版阶段

| 阶段 | Notebook | 内容 |
| --- | --- | --- |
| 00 | [数据准备](notebooks/zh-CN/00_data_preparation.ipynb) | 完全重复去重、有效性与历史计数审计、固定切分和特征缓存 |
| 01 | [探索性分析](notebooks/zh-CN/01_eda.ipynb) | 回忆率、复习间隔、历史练习、语言组合、用户和词元频率 |
| 02 | [HLR 基线](notebooks/zh-CN/02_hlr_baselines.ipynb) | HLR、HLR -lex、Train mean |
| 03 | [HLR 特征扩展](notebooks/zh-CN/03_hlr_experiments.ipynb) | HLR-User、HLR-Lang、HLR-User+Lang |
| 04 | [随机森林](notebooks/zh-CN/04_random_forest.ipynb) | RF-logh（E1）、RF-p（E2）、选参和全量 Train 拟合 |
| 05 | [LightGBM](notebooks/zh-CN/05_lightgbm.ipynb) | LGBM-logh（E3）、LGBM-p（E4）、选参和提前停止 |
| 06 | [神经 HLR](notebooks/zh-CN/06_neural_hlr.ipynb) | N-HLR、N-HLR -lex、HLR 初始化及 Validation 训练轨迹 |
| 07 | [模型比较](notebooks/zh-CN/07_model_comparison.ipynb) | 十二模型 Validation/Test 表、Test 预测及复习间隔诊断 |
| 08 | [非线性消融](notebooks/zh-CN/08_nonlinearity_ablation.ipynb) | 原始 HLR -lex、对齐后的线性 HLR 和四单元 ReLU N-HLR，三个配对种子 |
| 09 | [MAE 与 RMSE 分析](notebooks/zh-CN/09_mae_rmse_analysis.ipynb) | 标签与预测分布、极高预测占比、HLR 与 RF-p 的分组误差贡献 |
| 10 | [RF 与 LightGBM 相似性](notebooks/zh-CN/10_rf_lightgbm_similarity.ipynb) | 原始与最终预测配对、裁剪边界及预测重建核查 |

### 执行与自动生成文件

- 从 notebook 所在目录运行，原始文件放在 `data/raw/settles.acl16.learning_traces.13m.csv.gz`。
- 两种语言都使用项目根目录下的原始数据。共用训练代码默认显示英文，中文版通过 `Pipeline(language="zh-CN")` 选择中文，语言不参与模型缓存标识。
- 拆分版自动补齐缺失依赖。直接运行 07–10 可能触发上游模型训练，匹配的主模型产物从 `data/artifacts/` 复用。
- 完整 notebook 直接拟合十二个主模型，数据预处理和六次对齐消融支持复用本地缓存。没有消融缓存时，08 会分别使用 42、43、44 三个种子训练两种对齐结构。
- 清洗数据、特征和模型缓存、审计报告与图保存在 `data/processed/`、`data/interim/`、`data/artifacts/` 和 `reports/`。运行时自动创建所需目录，Git 忽略其中产物，notebook 内保存的表格与图仍可直接查看。
- 全量拟合比读取结果耗时更长，也需要更多内存。运行耗时、内存统计，以及部分浮点或排序指标可能在重复运行间略有差异。
