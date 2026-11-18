# Notebook 运行说明

从项目根目录启动 Jupyter，按文件名打开任意 notebook，选择 `data-science` 环境并运行全部单元。每本 notebook 都能从全新 kernel 开始运行，无需先执行其他 notebook。需要的中间数据和模型由 `experiment_pipeline.py` 自动读取或生成。

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
| `08_error_analysis.ipynb` | 误差分解与预测诊断 |

`legacy_full_pipeline.ipynb` 是拆分前完整 notebook 的归档。`pipeline_steps/` 按实验阶段保存可阅读的原计算代码，`experiment_pipeline.py` 管理依赖和缓存；用 `python build_split_notebooks.py` 可重新生成各 notebook 和计算代码副本。若修改了归档中的实验代码，应重新生成并重新运行受影响的实验。

模型权重、选参表、资源统计、Validation 结果和 Test 逐行预测保存在 `data/artifacts/`。文件名与清单由源数据、切分、有关代码和软件版本确定；不匹配时自动生成新结果。首次直接运行比较或误差分析 notebook 可能触发缺失模型的完整训练，耗时和磁盘占用较大。后续运行会读取已有结果。随机森林模型文件可能较大；运行前应确认磁盘空间。请只加载本项目自己生成的 `joblib` 文件。
