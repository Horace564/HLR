# Notebook 运行说明

从项目根目录启动 Jupyter，按文件名打开任意 notebook，选择 `data-science` 环境并运行全部单元。每本 notebook 都能从全新 kernel 开始运行，无需先执行其他 notebook。拆分版需要的中间数据和模型由 `experiment_pipeline.py` 自动读取或生成；standalone 版使用自身内嵌代码，详见下方说明。

| 文件 | 内容 |
| --- | --- |
| `00_data_preparation.ipynb` | 去重、数据审计、固定切分和特征缓存 |
| `01_eda.ipynb` | 去重数据的探索性分析 |
| `02_hlr_baselines.ipynb` | HLR、HLR -lex、Train mean |
| `03_hlr_experiments.ipynb` | 三个 HLR 类别特征扩展模型 |
| `04_random_forest.ipynb` | RF-logh、RF-p |
| `05_lightgbm.ipynb` | LGBM-logh、LGBM-p |
| `06_neural_hlr.ipynb` | N-HLR、N-HLR -lex |
| `07_model_comparison.ipynb` | 十二模型 Validation/Test 对照和复习间隔分析 |
| `08_nonlinearity_ablation.ipynb` | 对齐训练规则后的线性与非线性 HLR 消融 |
| `09_mae_rmse_analysis.ipynb` | 预测分布及 MAE/RMSE 排名反转的误差分解 |
| `10_rf_lightgbm_similarity.ipynb` | RF 与 LightGBM 的逐样本输出相似性诊断 |

`legacy_full_pipeline.ipynb` 是上述 **00–10 的 standalone 完整项目**：数据处理、模型定义、选参、训练和评估代码全部展开为普通代码单元，不导入项目内的 Python 模块，也不依赖其他 notebook。只需携带该 notebook、原始 gzip 数据（放在 `data/raw/`）及其首页列出的 Python 依赖，即可从全新 kernel 完整运行。十二个主模型会在 notebook 内从头训练；数据预处理和三种子消融仍支持缓存复用。模型结构、超参数、切分、随机种子和分析内容与拆分版一致。旧版重复的误差分析及未纳入拆分版的校准/风险排序分析不再保留。

拆分版使用 `pipeline_steps/` 的共用计算代码和 `experiment_pipeline.py` 的依赖与缓存管理。修改拆分版或共用代码后，用 `python build_legacy_notebook.py` 将当前实现逐字展开到 standalone notebook，用 `python build_legacy_notebook.py --check` 检查一致性。生成脚本只在同步内容时使用，运行 standalone notebook 无需携带它们。若在整合版中修改分析单元，可用 `python build_split_notebooks.py` 按单元来源导回全部 11 本拆分版（变化的 notebook 会清空旧输出）；`python build_split_notebooks.py --check` 只检查、不写文件。训练代码如需同步回拆分版，应先更新 `pipeline_steps/` 再重新生成整合版。这两个脚本都不会覆盖 `pipeline_steps/`，也不依赖旧版的固定单元序号。

拆分版的模型权重、选参表、资源统计、Validation 结果和 Test 逐行预测保存在 `data/artifacts/`。文件名与清单由源数据、切分、有关代码和软件版本确定；不匹配时自动生成新结果。首次直接运行比较或误差分析 notebook 可能触发缺失模型的完整训练，耗时和磁盘占用较大。后续运行会读取已有结果。standalone 版的十二个主模型直接保留在当前 kernel 中，不读取这些模型缓存。随机森林训练需要较多内存。请只加载本项目自己生成的 `joblib` 文件。
