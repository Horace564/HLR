# Checkpoint：全量 Train 的八模型对照

> 历史 checkpoint：以下描述对应拆分前的完整 notebook。原 `01_code.ipynb` 已归档为 `legacy_full_pipeline.ipynb`；当前运行入口见 `NOTEBOOKS.md`。

## 当前状态

Notebook `01_code.ipynb` 保留第 1–8 节。第 4 节为随机森林 E1/E2，第 5 节为 LightGBM E3/E4，第 6 节为两个神经半衰期模型，第 7 节统一比较 Validation/Test，第 8 节按 delta 比较 Test。原第 9 节（同一批 100 万行对照）和第 10 节（单独全量 E4）已删除。

四个树模型的**候选选参**仍使用相同的固定 20 万条 Train 行及完整 Validation；**正式训练**改为全部 9,255,042 条 Train。两个 HLR baseline 与两个神经模型原本已经使用全部 Train。因此第 7、8 节运行后，八模型的训练行数一致。以前 100 万行树模型的结果是历史结果，不能与新全量训练的结果混用；受影响的 Notebook 输出已清空，待重新运行。

## 数据与评估约定

- 原文件行顺序固定划分 Train / Validation / Test = 72% / 18% / 10%，不重新洗牌。选参只用 Train 子样本和完整 Validation；Test 仅用于最终评估。
- `x_correct=sqrt(1+history_correct)`，`x_wrong=sqrt(1+history_seen-history_correct)`，`bias=1`。E1/E3 用三项输入；E2/E4 再加入 `delta_days=delta/86400`。四个树模型都不含 lexeme。
- `p_target=clip(p_recall,0.0001,0.9999)`；`h_observed=clip(-delta_days/log2(p_target),15/1440,274)` 天。E1/E3 预测 `log2(h_observed)`，经 `2^score` 得 `h_hat`，再得 `p_hat`。E2/E4 直接拟合 `p_target`；其 `h_hat` 从 `p_hat` 与 `delta_days` 反推，只是诊断量。
- 当前缓存约 83.89% 的 `p_target` 位于上界、50.92% 的 `h_observed` 位于上界。报告整体 `Spearman(h)` 时也报告非边界子集 `MIN_H < h_observed < MAX_H` 且 `delta_days > 0` 的相关系数。
- 主指标为 `MAE(p)`。第 7 节另外报告 `RMSE`、`R²`、AUC、`Spearman(h)` 和样本数。第 8 节在 `[0,3)`、`[3,7)`、`[7,30)`、`[30,+∞)` 天四组报告 Test MAE、AUC、Spearman，并检查分组 MAE 的加权均值与总体一致。
- 全量特征矩阵直接创建为 `float32`；标签和评估计算维持原 Notebook 口径。压缩 CSV 放在项目的 `data/raw/`。本机使用已安装依赖的 `data-science` 环境运行；随机森林在 CPU 上训练。

## CP1 — E1：RF-logh

输入 `[x_correct,x_wrong,bias]`，目标 `log2(h_observed)`，平方误差回归。固定 `n_estimators=100`、`max_features=1.0`、`random_state=42`，六组候选为 `max_depth∈{8,12,16}` × `min_samples_leaf∈{50,100}`。在 20 万行选参后，用最佳组合在全部 Train 重训。候选块只输出六组 Validation `MAE(p)` 等选参信息；正式块只输出 E1 的训练时间、验证预测时间、RSS 和完整 Validation 指标。检查预测有限且位于裁剪范围。

## CP2 — E2：RF-p

输入 `[x_correct,x_wrong,bias,delta_days]`，目标 `p_target`，平方误差回归。使用与 CP1 相同的候选样本和六组 RF 参数，但独立按 Validation `MAE(p)` 选参。最佳组合在全部 Train 重训；对预测的 `p_hat` 裁剪后反推 `h_hat`。候选块只输出六组表现；正式块只输出 E2 资源与完整 Validation 指标。检查 `p_hat` 和 `h_hat` 的边界。

## CP3 — E3：LGBM-logh

输入和标签同 E1。固定 `n_estimators=1000`、`learning_rate=0.05`、`random_state=42`，六组候选为 `num_leaves∈{15,31,63}` × `min_child_samples∈{50,200}`。在相同 20 万行拟合，按完整 Validation 的 `log2(h_observed)` 均方误差提前停止 50 轮，再以 Validation `MAE(p)` 选配置。最佳配置在全部 Train 重训并重新提前停止。候选块只输出六组表现；正式块只输出 E3 资源、最佳轮数和自身 Validation 指标。

## CP4 — E4：LGBM-p

输入和标签同 E2，候选网格同 E3。按完整 Validation 的 `p_target` 均方误差提前停止 50 轮，再以 Validation `MAE(p)` 选配置。最佳配置在全部 Train 重训并重新提前停止。候选块只输出六组表现；正式块只输出 E4 资源、最佳轮数和自身 Validation 指标。E4 的半衰期指标仍是反推诊断量。

## 运行与验收

从头顺序运行 Notebook：生成缓存和固定切分，运行 baseline、E1–E4、两个 NN，然后运行第 7、8 节。确认第 7 节两张表各有八行且 `n_train=9,255,042`；第 8 节每个指标表均为八模型 × 四区间。记录各模型训练时间和峰值 RSS。本机全量 RF 的耗时与内存仍需实测；运行前关闭其他占用内存的程序，并留意可用内存。
