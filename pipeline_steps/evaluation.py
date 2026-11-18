"""Original experiment code, executed by experiment_pipeline.py."""

# === ORIGINAL CELL 083 ===
def predict_hlr_baseline_batch(weights, batch):
    score = (weights[0] * batch["x_correct"]
             + weights[1] * batch["x_wrong"] + weights[2])
    if len(weights) > 3:
        columns = batch["lexeme_column"]
        known = (columns >= 3) & (columns < len(weights))
        score[known] += weights[columns[known]]
    h_pred = np.exp2(np.clip(score, LOG_MIN_H, LOG_MAX_H))
    p_pred = np.clip(np.exp2(-batch["delta_days"] / h_pred), MIN_P, MAX_P)
    return p_pred, h_pred

def final_metric_row(name, n_train, split, p_true_split, h_true_split,
                     days_split, p_pred, h_pred):
    error = p_pred - p_true_split
    centered = p_true_split - float(np.mean(p_true_split))
    sse = float(np.dot(error, error))
    sst = float(np.dot(centered, centered))
    interior = ((h_true_split > MIN_H) & (h_true_split < MAX_H)
                & (days_split > 0))
    n_interior = int(interior.sum())
    return {
        "model": name, "split": split, "n_train": n_train,
        "n_eval": len(p_true_split),
        "MAE(p)": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(sse / len(error))),
        "R²": 1.0 - sse / sst if sst > 0 else np.nan,
        "AUC": auc_fixed(np.rint(p_true_split), p_pred),
        "Spearman(h)": spearman_with_ties(h_true_split, h_pred),
        "n_spearman_h": len(p_true_split),
        "n_h_interior": n_interior,
        "h_interior_share": n_interior / len(p_true_split),
        "Spearman(h)_interior": spearman_with_ties(
            h_true_split[interior], h_pred[interior]
        ) if n_interior > 1 else np.nan,
    }

def mean_final_metric_row(split_name, p_true_split):
    count = len(p_true_split)
    error = mean_constant_p - p_true_split
    sse = float(np.dot(error, error))
    centered = p_true_split - float(np.mean(p_true_split))
    sst = float(np.dot(centered, centered))
    binary = np.rint(p_true_split)
    auc = 0.5 if np.any(binary == 0) and np.any(binary == 1) else np.nan
    return {
        "model": "Train mean", "split": split_name,
        "n_train": train_end, "n_eval": count,
        "MAE(p)": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(sse / count)),
        "R²": 1.0 - sse / sst if sst > 0 else np.nan,
        "AUC": auc, "Spearman(h)": np.nan,
        "n_spearman_h": 0, "n_h_interior": 0,
        "h_interior_share": np.nan,
        "Spearman(h)_interior": np.nan,
    }

model_order = [
    "HLR", "HLR -lex", "Train mean", "HLR-User", "HLR-Lang", "HLR-User+Lang",
    "N-HLR", "N-HLR -lex", "RF-logh (E1)", "RF-p (E2)",
    "LGBM-logh (E3)", "LGBM-p (E4)",
]
n_train_by_model = {name: train_end for name in model_order}

# Summarize all twelve models on the same Validation split.
val_start, val_stop = splits["Validation"]
n_val = val_stop - val_start
p_true = records["p_target"][val_start:val_stop]
h_true = records["h_observed"][val_start:val_stop]
days = records["delta_days"][val_start:val_stop]
val_p_variance = float(np.var(p_true))
validation_rows = []
for name, weights, prior in (
    ("HLR", hlr_weights, hlr_results),
    ("HLR -lex", hlr_lexless_weights, hlr_lexless_results),
):
    p_pred = np.empty(n_val, dtype=np.float64)
    h_pred = np.empty(n_val, dtype=np.float64)
    for left in range(val_start, val_stop, RF_BATCH):
        right = min(left + RF_BATCH, val_stop)
        p_batch, h_batch = predict_hlr_baseline_batch(weights, records[left:right])
        p_pred[left-val_start:right-val_start] = p_batch
        h_pred[left-val_start:right-val_start] = h_batch
    assert np.isclose(np.abs(p_pred - p_true).mean(), float(prior.loc[
        prior["split"] == "Validation", "MAE(p)"
    ].iloc[0]))
    validation_rows.append(final_metric_row(
        name, train_end, "Validation", p_true, h_true, days, p_pred, h_pred
    ))

validation_rows.append(mean_final_metric_row("Validation", p_true))

for frame in (hlr_user_validation, hlr_lang_validation, hlr_both_validation):
    validation_rows.append(frame.iloc[0].to_dict())

for name, p_pred, h_pred in (
    ("N-HLR", nn_hlr_val_p_pred, nn_hlr_val_h_pred),
    ("N-HLR -lex", nn_hlr_lexless_val_p_pred, nn_hlr_lexless_val_h_pred),
):
    validation_rows.append(final_metric_row(
        name, train_end, "Validation", p_true, h_true, days, p_pred, h_pred
    ))

for frame in (rf_e1_validation, rf_e2_validation,
              lgbm_e3_validation, lgbm_e4_validation):
    row = frame.iloc[0]
    validation_rows.append({
        "model": row["model"], "split": "Validation",
        "n_train": int(row["n_train"]), "n_eval": int(row["n_validation"]),
        "MAE(p)": row["MAE(p)"],
        "RMSE": row["RMSE(p)"],
        "R²": 1.0 - float(row["RMSE(p)"]) ** 2 / val_p_variance,
        **{key: row[key] for key in (
            "AUC", "Spearman(h)",
            "n_spearman_h", "n_h_interior", "h_interior_share",
            "Spearman(h)_interior",
        )},
    })

test_start_row, test_stop_row = splits["Test"]
n_test = test_stop_row - test_start_row
test_true_p = records["p_target"][test_start_row:test_stop_row]
test_true_h = records["h_observed"][test_start_row:test_stop_row]
test_days = records["delta_days"][test_start_row:test_stop_row]
test_predictions = {
    name: (np.empty(n_test, dtype=np.float64), np.empty(n_test, dtype=np.float64))
    for name in model_order
}
for left in range(test_start_row, test_stop_row, RF_BATCH):
    right = min(left + RF_BATCH, test_stop_row)
    batch = records[left:right]
    offset_left, offset_right = left - test_start_row, right - test_start_row
    x3 = rf_x_from_records(batch)
    x4 = rf_p_x_from_records(batch)
    predictions = {
        "HLR": predict_hlr_baseline_batch(hlr_weights, batch),
        "HLR -lex": predict_hlr_baseline_batch(hlr_lexless_weights, batch),
        "Train mean": (
            np.full(len(batch), mean_constant_p, dtype=np.float64),
            np.full(len(batch), np.nan, dtype=np.float64),
        ),
        "HLR-User": predict_hlr_extended_batch(
            hlr_user_weights, hlr_user_layout, batch, categorical_records[left:right]
        ),
        "HLR-Lang": predict_hlr_extended_batch(
            hlr_lang_weights, hlr_lang_layout, batch, categorical_records[left:right]
        ),
        "HLR-User+Lang": predict_hlr_extended_batch(
            hlr_both_weights, hlr_both_layout, batch, categorical_records[left:right]
        ),
        "N-HLR": nn_hlr_model.predict(batch),
        "N-HLR -lex": nn_hlr_lexless_model.predict(batch),
        "RF-logh (E1)": rf_logh_predictions(rf_e1_model, batch),
        "RF-p (E2)": rf_p_predictions(rf_e2_model, batch),
        "LGBM-logh (E3)": lgbm_logh_predictions(
            lgbm_e3_model, x3, batch["delta_days"]
        ),
        "LGBM-p (E4)": lgbm_p_predictions(
            lgbm_e4_model, x4, batch["delta_days"]
        ),
    }
    for name, (p_batch, h_batch) in predictions.items():
        p_all, h_all = test_predictions[name]
        p_all[offset_left:offset_right] = p_batch
        h_all[offset_left:offset_right] = h_batch

test_rows = []
for name in model_order:
    p_pred, h_pred = test_predictions[name]
    assert np.isfinite(p_pred).all()
    if name != "Train mean":
        assert np.isfinite(h_pred).all()
    assert np.all((MIN_P <= p_pred) & (p_pred <= MAX_P))
    if name == "Train mean":
        test_rows.append(mean_final_metric_row("Test", test_true_p))
    else:
        # exp2(log2(bound)) may be a few floating-point ulps outside the bound.
        h_tolerance = 1e-10
        assert np.all((MIN_H - h_tolerance <= h_pred) &
                      (h_pred <= MAX_H + h_tolerance)), (
            name, float(np.min(h_pred)), float(np.max(h_pred))
        )
        test_rows.append(final_metric_row(
            name, n_train_by_model[name], "Test", test_true_p, test_true_h,
            test_days, p_pred, h_pred,
        ))
final_summary = pd.DataFrame(validation_rows + test_rows)
final_summary["model"] = pd.Categorical(
    final_summary["model"], categories=model_order, ordered=True
)
final_summary["split"] = pd.Categorical(
    final_summary["split"], categories=["Validation", "Test"], ordered=True
)
final_summary = final_summary.sort_values(["model", "split"]).reset_index(drop=True)
for name, result in (("HLR", hlr_results), ("HLR -lex", hlr_lexless_results)):
    old_mae = float(result.loc[result["split"] == "Test", "MAE(p)"].iloc[0])
    new_mae = float(final_summary.loc[
        (final_summary["model"] == name) & (final_summary["split"] == "Test"),
        "MAE(p)",
    ].iloc[0])
    assert np.isclose(old_mae, new_mae)
validation_summary = final_summary[final_summary["split"] == "Validation"].copy()
test_summary = final_summary[final_summary["split"] == "Test"].copy()
assert len(validation_summary) == len(test_summary) == 12
display(validation_summary.round(6).reset_index(drop=True))
display(test_summary.round(6).reset_index(drop=True))

# === ORIGINAL CELL 086 ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

task92_needed = ["test_true_p", "test_predictions", "MIN_P", "MAX_P"]
task92_missing = [name for name in task92_needed if name not in globals()]
if task92_missing:
    raise RuntimeError(
        "Section 9.1 needs the current kernel's section 8 predictions: "
        + ", ".join(task92_missing)
    )

task92_models = ["HLR", "N-HLR", "RF-p (E2)", "LGBM-p (E4)"]
task92_true = np.asarray(test_true_p, dtype=np.float64)
task92_n = len(task92_true)
task92_series = {"p_target (true)": task92_true}
for name in task92_models:
    task92_series[name] = np.asarray(test_predictions[name][0], dtype=np.float64)
assert all(len(values) == task92_n and np.isfinite(values).all()
           for values in task92_series.values())
assert all(np.all((MIN_P <= values) & (values <= MAX_P))
           for values in task92_series.values())

task92_distribution_rows = []
task92_edges = np.linspace(MIN_P, MAX_P, 51)
task92_histograms = {}
for name, values in task92_series.items():
    p10, p50, p90 = np.quantile(values, [0.10, 0.50, 0.90])
    task92_distribution_rows.append({
        "series": name, "n_test": task92_n,
        "mean": float(values.mean()), "std": float(values.std()),
        "P10": float(p10), "P50": float(p50), "P90": float(p90),
        "share_ge_0.95": float(np.mean(values >= 0.95)),
        "share_at_MIN_P": float(np.mean(values == MIN_P)),
        "share_at_MAX_P": float(np.mean(values == MAX_P)),
    })
    task92_histograms[name] = np.histogram(values, bins=task92_edges)[0] / task92_n

task92_distribution = pd.DataFrame(task92_distribution_rows)
display(task92_distribution.round(6))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
colors = ["black", "#4C78A8", "#54A24B", "#F58518", "#E45756"]
for (name, masses), color in zip(task92_histograms.items(), colors):
    for ax in axes:
        ax.stairs(masses, task92_edges, label=name, linewidth=1.7, color=color)
axes[0].set(
    title="Full Test distribution", xlim=(0, 1),
    xlabel="p_target or predicted p", ylabel="share of all Test rows / bin",
)
axes[0].set_yscale("log")
axes[1].set(
    title="Interior zoom (p < 0.95)", xlim=(0, 0.95),
    xlabel="p_target or predicted p", ylabel="share of all Test rows / bin",
)
for ax in axes:
    ax.grid(alpha=0.2)
axes[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
plt.show()

# === ORIGINAL CELL 088 ===
task92_group_names = [
    "p_target=MIN_P", "MIN_P<p<=0.2", "0.2<p<=0.5",
    "0.5<p<MAX_P", "p_target=MAX_P",
]
task92_group_masks = [
    task92_true == MIN_P,
    (task92_true > MIN_P) & (task92_true <= 0.2),
    (task92_true > 0.2) & (task92_true <= 0.5),
    (task92_true > 0.5) & (task92_true < MAX_P),
    task92_true == MAX_P,
]
task92_covered = np.zeros(task92_n, dtype=np.int8)
for mask in task92_group_masks:
    task92_covered += mask
assert np.all(task92_covered == 1), "p_target groups must cover Test exactly once"

task92_decomposition_rows = []
task92_check_rows = []
for model in task92_models:
    error = task92_series[model] - task92_true
    abs_error = np.abs(error)
    squared_error = error ** 2
    for group, mask in zip(task92_group_names, task92_group_masks):
        n_group = int(mask.sum())
        if not n_group:
            raise ValueError(f"Empty p_target group: {group}")
        task92_decomposition_rows.append({
            "group": group, "n_group": n_group,
            "share": n_group / task92_n, "model": model,
            "MAE_g": float(abs_error[mask].mean()),
            "MSE_g": float(squared_error[mask].mean()),
            "C_MAE": float(abs_error[mask].sum() / task92_n),
            "C_MSE": float(squared_error[mask].sum() / task92_n),
        })
    rows = [row for row in task92_decomposition_rows if row["model"] == model]
    mae = float(abs_error.mean())
    mse = float(squared_error.mean())
    sum_mae = sum(row["C_MAE"] for row in rows)
    sum_mse = sum(row["C_MSE"] for row in rows)
    assert np.isclose(mae, sum_mae, rtol=0, atol=1e-12)
    assert np.isclose(mse, sum_mse, rtol=0, atol=1e-12)
    task92_check_rows.append({
        "model": model, "overall_MAE": mae, "sum_C_MAE": sum_mae,
        "overall_RMSE": float(np.sqrt(mse)),
        "sqrt_sum_C_MSE": float(np.sqrt(sum_mse)),
    })

task92_decomposition = pd.DataFrame(task92_decomposition_rows)
task92_checks = pd.DataFrame(task92_check_rows)
display(task92_decomposition.round(6))
display(task92_checks.round(6))

task92_main_rows = []
for group in task92_group_names:
    hlr = task92_decomposition.loc[
        (task92_decomposition["group"] == group)
        & (task92_decomposition["model"] == "HLR")
    ].iloc[0]
    e4 = task92_decomposition.loc[
        (task92_decomposition["group"] == group)
        & (task92_decomposition["model"] == "LGBM-p (E4)")
    ].iloc[0]
    task92_main_rows.append({
        "group": group, "n_group": int(hlr["n_group"]),
        "share": float(hlr["share"]),
        "HLR_MAE_g": float(hlr["MAE_g"]),
        "E4_MAE_g": float(e4["MAE_g"]),
        "HLR_C_MAE": float(hlr["C_MAE"]),
        "E4_C_MAE": float(e4["C_MAE"]),
        "delta_C_MAE_HLR_minus_E4": float(hlr["C_MAE"] - e4["C_MAE"]),
        "HLR_C_MSE": float(hlr["C_MSE"]),
        "E4_C_MSE": float(e4["C_MSE"]),
        "delta_C_MSE_HLR_minus_E4": float(hlr["C_MSE"] - e4["C_MSE"]),
    })
task92_main = pd.DataFrame(task92_main_rows)
display(task92_main.round(6))

x = np.arange(len(task92_group_names))
width = 0.36
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, metric, title in zip(
    axes, ["C_MAE", "C_MSE"],
    ["Contribution to overall MAE", "Contribution to overall MSE"],
):
    hlr_values = task92_decomposition.loc[
        task92_decomposition["model"] == "HLR", metric
    ].to_numpy()
    e4_values = task92_decomposition.loc[
        task92_decomposition["model"] == "LGBM-p (E4)", metric
    ].to_numpy()
    ax.bar(x - width / 2, hlr_values, width, label="HLR", color="#4C78A8")
    ax.bar(x + width / 2, e4_values, width,
           label="LGBM-p (E4)", color="#E45756")
    ax.set(title=title, xticks=x,
           xticklabels=["MIN_P", "(MIN,.2]", "(.2,.5]", "(.5,MAX)", "MAX_P"])
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
fig.tight_layout()
plt.show()

# === ORIGINAL CELL 090 ===
task92_top_k = int(np.ceil(task92_n * 0.01))
task92_tail_rows = []
for model in task92_models:
    error = task92_series[model] - task92_true
    abs_error = np.abs(error)
    squared_error = error ** 2
    p90, p99 = np.quantile(abs_error, [0.90, 0.99])
    top_idx = np.argpartition(squared_error, task92_n - task92_top_k)[
        task92_n - task92_top_k:
    ]
    task92_tail_rows.append({
        "model": model, "top_k": task92_top_k,
        "P90_abs_error": float(p90), "P99_abs_error": float(p99),
        "own_worst_1pct_share_of_total_MSE": float(
            squared_error[top_idx].sum() / squared_error.sum()
        ),
    })
task92_tail = pd.DataFrame(task92_tail_rows)
display(task92_tail.round(6))

hlr_abs_error = np.abs(task92_series["HLR"] - task92_true)
hlr_worst_idx = np.argpartition(hlr_abs_error, task92_n - task92_top_k)[
    task92_n - task92_top_k:
]
task92_fixed_rows = []
for model in task92_models:
    error = task92_series[model] - task92_true
    squared_error = error ** 2
    task92_fixed_rows.append({
        "model": model, "fixed_HLR_worst_rows": task92_top_k,
        "MAE_on_rows": float(np.abs(error[hlr_worst_idx]).mean()),
        "MSE_on_rows": float(squared_error[hlr_worst_idx].mean()),
        "share_of_model_total_MSE": float(
            squared_error[hlr_worst_idx].sum() / squared_error.sum()
        ),
    })
task92_fixed_hlr_worst = pd.DataFrame(task92_fixed_rows)
display(task92_fixed_hlr_worst.round(6))

task92_worst_group = pd.DataFrame([
    {
        "group": name,
        "rows_in_HLR_worst_1pct": int(mask[hlr_worst_idx].sum()),
        "share_within_HLR_worst_1pct": float(mask[hlr_worst_idx].mean()),
    }
    for name, mask in zip(task92_group_names, task92_group_masks)
])
display(task92_worst_group.round(6))

# === ORIGINAL CELL 094 ===
import matplotlib.pyplot as plt

delta_model_order = model_order  # 第 8 节的十二个模型，顺序保持一致
labels = ["[0, 3) 天", "[3, 7) 天", "[7, 30) 天", "[30, +∞) 天"]
test_group_ids = np.searchsorted(np.array([3.0, 7.0, 30.0]),
                                 test_days, side="right")
test_delta_rows = []
for name in delta_model_order:
    p_pred, h_pred = test_predictions[name]
    for group_id, label in enumerate(labels):
        mask = test_group_ids == group_id
        test_delta_rows.append({
            "model": name, "delta 区间": label, "rows": int(mask.sum()),
            "MAE(p)": float(np.abs(p_pred[mask] - test_true_p[mask]).mean()),
            "AUC": auc_fixed(np.rint(test_true_p[mask]), p_pred[mask]),
            "Spearman(h)": (np.nan if name == "Train mean" else
                             spearman_with_ties(test_true_h[mask], h_pred[mask])),
        })
final_test_delta = pd.DataFrame(test_delta_rows)
for name in delta_model_order:
    part = final_test_delta[final_test_delta["model"] == name]
    overall = float(final_summary.loc[
        (final_summary["model"] == name) & (final_summary["split"] == "Test"),
        "MAE(p)",
    ].iloc[0])
    assert part["rows"].sum() == n_test
    assert np.isclose(np.average(part["MAE(p)"], weights=part["rows"]), overall)


delta_order = ["[0, 3) 天", "[3, 7) 天", "[7, 30) 天", "[30, +∞) 天"]
delta_counts = (final_test_delta[final_test_delta["model"] == "HLR"]
                .set_index("delta 区间").loc[delta_order, "rows"])
print("Test 各 delta 区间的样本数；十二个模型共用这些行。")
display(delta_counts.rename("rows").to_frame())

delta_matrices = {}
for metric in ("MAE(p)", "AUC", "Spearman(h)"):
    matrix = (final_test_delta.pivot(index="model", columns="delta 区间",
                                     values=metric)
              .loc[delta_model_order, delta_order])
    assert matrix.shape == (12, 4)
    if metric == "Spearman(h)":
        assert matrix.loc["Train mean"].isna().all()
        assert np.isfinite(matrix.drop(index="Train mean").to_numpy()).all()
    else:
        assert np.isfinite(matrix.to_numpy()).all()
    delta_matrices[metric] = matrix
    print(f"Test {metric}：逐行比较十二个模型；逐列比较同一复习间隔。")
    display(matrix.round(6))

plot_labels = ["0–<3d", "3–<7d", "7–<30d", "≥30d"]
colors = plt.get_cmap("tab20").colors
markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "d"]
line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
x_centers = np.arange(4)
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharex=True)
for model_index, model_name in enumerate(delta_model_order):
    # 只横向错开绘图位置；每个点的 y 值仍是表中的真实指标。
    x_positions = x_centers + (model_index - 5.5) * 0.032
    for ax, metric in zip(axes, ("MAE(p)", "AUC", "Spearman(h)")):
        if model_name == "Train mean" and metric == "Spearman(h)":
            continue
        ax.plot(x_positions, delta_matrices[metric].loc[model_name].to_numpy(),
                marker=markers[model_index], linestyle=line_styles[model_index],
                linewidth=1.8, markersize=6, markeredgecolor="white",
                markeredgewidth=0.7, color=colors[model_index],
                label=model_name)
for ax, metric in zip(axes, ("MAE(p)", "AUC", "Spearman(h)")):
    ax.set_title(metric)
    ax.set_xticks(x_centers, plot_labels)
    ax.set_xlim(-0.25, 3.25)
    ax.set_xlabel("delta (days)")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("lower is better")
axes[1].set_ylabel("higher is better")
axes[2].set_ylabel("higher is better")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, 0.02), frameon=False)
fig.suptitle('Twelve-model Test performance by review interval')
fig.subplots_adjust(bottom=0.34, wspace=0.28, top=0.86)
plt.show()

# === ORIGINAL CELL 098 ===
import numpy as np
import pandas as pd
from IPython.display import display

diag_required = [
    "test_predictions", "records", "splits", "RF_BATCH", "MIN_P", "MAX_P",
    "LOG_MIN_H", "LOG_MAX_H", "rf_x_from_records", "rf_p_x_from_records",
    "rf_e1_model", "rf_e2_model", "lgbm_e3_model", "lgbm_e4_model",
    "spearman_with_ties",
]
diag_missing = [name for name in diag_required if name not in globals()]
if diag_missing:
    raise RuntimeError(
        "Missing variables from the current kernel: " + ", ".join(diag_missing)
        + ". This section needs the trained models and section 8 test_predictions."
    )

diag_test_start, diag_test_stop = splits["Test"]
diag_n_test = diag_test_stop - diag_test_start

def diag_metric_row(pair, space, rf_values, lgb_values):
    rf_values = np.asarray(rf_values, dtype=np.float64)
    lgb_values = np.asarray(lgb_values, dtype=np.float64)
    assert rf_values.shape == lgb_values.shape == (diag_n_test,)
    assert np.isfinite(rf_values).all() and np.isfinite(lgb_values).all()
    d = rf_values - lgb_values
    abs_d = np.abs(d)
    p50, p90, p99 = np.quantile(abs_d, [0.50, 0.90, 0.99])
    pearson = (float(np.corrcoef(rf_values, lgb_values)[0, 1])
               if np.std(rf_values) > 0 and np.std(lgb_values) > 0
               else np.nan)
    return {
        "pair": pair, "space": space, "n_test": diag_n_test,
        "Pearson": pearson,
        "Spearman": spearman_with_ties(rf_values, lgb_values),
        "mean_d": float(d.mean()),
        "mean_abs_d": float(abs_d.mean()),
        "P50_abs_d": float(p50), "P90_abs_d": float(p90),
        "P99_abs_d": float(p99),
        "share_abs_lt_1e-6": float(np.mean(abs_d < 1e-6)),
        "share_exact": float(np.mean(d == 0)),
    }

diag_pairs = [
    ("E1 vs E3", "RF-logh (E1)", "LGBM-logh (E3)",
     rf_e1_model, lgbm_e3_model, rf_x_from_records, "raw_log2_h", True),
    ("E2 vs E4", "RF-p (E2)", "LGBM-p (E4)",
     rf_e2_model, lgbm_e4_model, rf_p_x_from_records, "raw_p", False),
]
diag_stat_rows = []
diag_boundary_rows = []
diag_check_rows = []

for pair, rf_name, lgb_name, rf_model, lgb_model, make_x, raw_space, predicts_logh in diag_pairs:
    rf_p = np.asarray(test_predictions[rf_name][0], dtype=np.float64)
    lgb_p = np.asarray(test_predictions[lgb_name][0], dtype=np.float64)
    assert rf_p.shape == lgb_p.shape == (diag_n_test,)
    diag_stat_rows.append(diag_metric_row(pair, "clipped_p", rf_p, lgb_p))

    same_bound = ((rf_p == MIN_P) & (lgb_p == MIN_P)) | (
        (rf_p == MAX_P) & (lgb_p == MAX_P)
    )
    both_interior = ((MIN_P < rf_p) & (rf_p < MAX_P)
                     & (MIN_P < lgb_p) & (lgb_p < MAX_P))
    diag_boundary_rows.append({
        "pair": pair,
        "same_p_bound_share": float(same_bound.mean()),
        "n_both_interior": int(both_interior.sum()),
        "share_abs_lt_1e-6_interior": (
            float(np.mean(np.abs(rf_p[both_interior] - lgb_p[both_interior]) < 1e-6))
            if both_interior.any() else np.nan
        ),
    })

    rf_raw = np.empty(diag_n_test, dtype=np.float64)
    lgb_raw = np.empty(diag_n_test, dtype=np.float64)
    rf_reconstruction_max = 0.0
    lgb_reconstruction_max = 0.0
    for left in range(diag_test_start, diag_test_stop, RF_BATCH):
        right = min(left + RF_BATCH, diag_test_stop)
        batch = records[left:right]
        x = make_x(batch)
        lo, hi = left - diag_test_start, right - diag_test_start
        rf_chunk = rf_model.predict(x)
        lgb_chunk = lgb_model.predict(x, num_iteration=lgb_model.best_iteration_)
        rf_raw[lo:hi] = rf_chunk
        lgb_raw[lo:hi] = lgb_chunk

        if predicts_logh:
            rf_h = np.exp2(np.clip(rf_chunk, LOG_MIN_H, LOG_MAX_H))
            lgb_h = np.exp2(np.clip(lgb_chunk, LOG_MIN_H, LOG_MAX_H))
            rf_rebuilt = np.clip(
                np.exp2(-batch["delta_days"] / rf_h), MIN_P, MAX_P
            )
            lgb_rebuilt = np.clip(
                np.exp2(-batch["delta_days"] / lgb_h), MIN_P, MAX_P
            )
        else:
            rf_rebuilt = np.clip(rf_chunk, MIN_P, MAX_P)
            lgb_rebuilt = np.clip(lgb_chunk, MIN_P, MAX_P)
        rf_reconstruction_max = max(
            rf_reconstruction_max, float(np.max(np.abs(rf_rebuilt - rf_p[lo:hi])))
        )
        lgb_reconstruction_max = max(
            lgb_reconstruction_max, float(np.max(np.abs(lgb_rebuilt - lgb_p[lo:hi])))
        )

    diag_stat_rows.append(diag_metric_row(pair, raw_space, rf_raw, lgb_raw))
    diag_check_rows.append({
        "pair": pair,
        "max_rebuild_error_RF": rf_reconstruction_max,
        "max_rebuild_error_LGBM": lgb_reconstruction_max,
    })
    del rf_raw, lgb_raw, same_bound, both_interior

diag_stats = pd.DataFrame(diag_stat_rows)
diag_boundary = pd.DataFrame(diag_boundary_rows)
diag_reconstruction = pd.DataFrame(diag_check_rows)
assert (diag_reconstruction[["max_rebuild_error_RF",
                             "max_rebuild_error_LGBM"]].to_numpy() < 1e-10).all()
with pd.option_context("display.float_format", lambda value: f"{value:.6g}"):
    display(diag_stats)
    display(diag_boundary)
    display(diag_reconstruction)

# === ORIGINAL CELL 099 ===
import matplotlib.pyplot as plt

diag_scatter_n = min(20_000, diag_n_test)
diag_sample = np.random.default_rng(20261115).choice(
    diag_n_test, size=diag_scatter_n, replace=False
)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for row, (pair, rf_name, lgb_name, *_rest) in enumerate(diag_pairs):
    rf_p = test_predictions[rf_name][0]
    lgb_p = test_predictions[lgb_name][0]
    ax_scatter, ax_hist = axes[row]
    ax_scatter.scatter(
        rf_p[diag_sample], lgb_p[diag_sample],
        s=7, alpha=0.18, linewidths=0, rasterized=True,
    )
    ax_scatter.plot([MIN_P, MAX_P], [MIN_P, MAX_P], "k--", linewidth=1)
    ax_scatter.set(xlim=(0, 1), ylim=(0, 1),
                   xlabel="RF p_hat", ylabel="LightGBM p_hat",
                   title=f"{pair}: sampled Test predictions (n={diag_scatter_n:,})")
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.grid(alpha=0.2)

    abs_d = np.abs(rf_p - lgb_p)
    positive = abs_d[abs_d > 0]
    ax_hist.hist(np.maximum(positive, 1e-12),
                 bins=np.logspace(-12, 0, 75),
                 color="#4C78A8", edgecolor="none")
    ax_hist.set_xscale("log")
    ax_hist.set(xlim=(1e-12, 1), xlabel="|p_RF - p_LightGBM|",
                ylabel="Test row count",
                title=f"{pair}: all Test rows; exact zero = {(abs_d == 0).mean():.2%}")
    ax_hist.grid(alpha=0.2)

fig.suptitle("RF vs LightGBM: paired prediction diagnostics", fontsize=15)
fig.tight_layout()
plt.show()

# === ORIGINAL CELL 102 ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

task93_needed = [
    "records", "train_end", "test_true_p", "test_days",
    "test_predictions", "auc_fixed", "MIN_P", "MAX_P",
]
task93_missing = [name for name in task93_needed if name not in globals()]
if task93_missing:
    raise RuntimeError(
        "Section 11.2 needs the current kernel's section 8 results: "
        + ", ".join(task93_missing)
    )

task93_model_names = ["HLR", "N-HLR", "RF-p (E2)", "LGBM-p (E4)"]
task93_true = np.asarray(test_true_p, dtype=np.float64)
task93_days = np.asarray(test_days, dtype=np.float64)
task93_n = len(task93_true)
task93_train_mean = float(records["p_target"][:train_end].mean())
task93_predictions = {
    name: np.asarray(test_predictions[name][0], dtype=np.float64)
    for name in task93_model_names
}
task93_predictions["Train mean"] = np.full(
    task93_n, task93_train_mean, dtype=np.float64
)
assert len(task93_days) == task93_n
assert all(len(pred) == task93_n and np.isfinite(pred).all()
           for pred in task93_predictions.values())
assert all(np.all((MIN_P <= pred) & (pred <= MAX_P))
           for pred in task93_predictions.values())

task93_edges = np.linspace(0.0, 1.0, 11)
task93_bin_labels = [
    f"[{task93_edges[b]:.1f}, {task93_edges[b+1]:.1f})"
    if b < 9 else "[0.9, 1.0]"
    for b in range(10)
]
task93_binary_recall = np.rint(task93_true)
task93_cal_rows = []
task93_summary_rows = []
for name, pred in task93_predictions.items():
    bin_id = np.clip(
        np.searchsorted(task93_edges, pred, side="right") - 1, 0, 9
    )
    ece = 0.0
    for b, label in enumerate(task93_bin_labels):
        mask = bin_id == b
        n_bin = int(mask.sum())
        mean_pred = float(pred[mask].mean()) if n_bin else np.nan
        mean_true = float(task93_true[mask].mean()) if n_bin else np.nan
        gap = mean_true - mean_pred if n_bin else np.nan
        if n_bin:
            ece += (n_bin / task93_n) * abs(gap)
        task93_cal_rows.append({
            "model": name, "pred_bin": label, "n_bin": n_bin,
            "share": n_bin / task93_n,
            "mean_pred": mean_pred, "mean_true": mean_true,
            "calibration_gap_true_minus_pred": gap,
        })
    error = pred - task93_true
    mse = float(np.mean(error ** 2))
    task93_summary_rows.append({
        "model": name, "n_test": task93_n,
        "mean_pred": float(pred.mean()), "std_pred": float(pred.std()),
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse, "RMSE": float(np.sqrt(mse)),
        "ECE10": float(ece),
        "AUC": auc_fixed(task93_binary_recall, pred),
    })

task93_calibration = pd.DataFrame(task93_cal_rows)
task93_summary = pd.DataFrame(task93_summary_rows)
task93_counts = task93_calibration.pivot(
    index="pred_bin", columns="model", values="n_bin"
).loc[task93_bin_labels, list(task93_predictions)]
print(f"Test mean p_target = {task93_true.mean():.6f}; "
      f"Train mean constant = {task93_train_mean:.6f}")
display(task93_summary.round(6))
display(task93_counts)
display(task93_calibration.round(6))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756", "#777777"]
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
for (name, color) in zip(task93_predictions, colors):
    part = task93_calibration[
        (task93_calibration["model"] == name)
        & (task93_calibration["n_bin"] > 0)
    ]
    axes[0].plot(
        part["mean_pred"], part["mean_true"],
        color=color, linewidth=1.3, alpha=0.8,
    )
    axes[0].scatter(
        part["mean_pred"], part["mean_true"],
        s=24 + 180 * np.sqrt(part["share"]),
        color=color, alpha=0.85, label=name,
    )
    occupied = part["pred_bin"].map(
        {label: i for i, label in enumerate(task93_bin_labels)}
    ).to_numpy()
    axes[1].plot(
        occupied + 0.5, part["n_bin"], marker="o",
        color=color, linewidth=1.3, label=name,
    )
axes[0].set(
    title="Reliability diagram (marker size reflects bin share)",
    xlim=(0, 1), ylim=(0, 1),
    xlabel="mean predicted p in bin",
    ylabel="mean observed p_target in bin",
)
axes[1].set(
    title="Test rows in each prediction bin",
    xticks=np.arange(10) + 0.5,
    xticklabels=[f"{b/10:.1f}" for b in range(10)],
    xlabel="predicted p bin (left edge)", ylabel="n_bin",
)
axes[1].set_yscale("log")
for ax in axes:
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
plt.show()

# === ORIGINAL CELL 104 ===
task93_risk = 1.0 - task93_true
task93_base_risk = float(task93_risk.mean())
assert task93_base_risk > 0
task93_curve_rows = []
task93_selected_rows = []
task93_curve_percents = np.arange(1, 51)

for name in task93_model_names:
    order = np.argsort(task93_predictions[name], kind="stable")
    cumulative_risk = np.cumsum(task93_risk[order], dtype=np.float64)
    for percent in task93_curve_percents:
        k = int(np.ceil(task93_n * percent / 100))
        selected_risk = float(cumulative_risk[k - 1] / k)
        row = {
            "model": name, "K_percent": int(percent), "n_selected": k,
            "Risk_at_K": selected_risk,
            "Lift_at_K": selected_risk / task93_base_risk,
        }
        task93_curve_rows.append(row)
        if percent in (5, 10, 20):
            task93_selected_rows.append(row)

for percent in (5, 10, 20):
    task93_selected_rows.append({
        "model": "Random expected", "K_percent": percent,
        "n_selected": int(np.ceil(task93_n * percent / 100)),
        "Risk_at_K": task93_base_risk, "Lift_at_K": 1.0,
    })
task93_curve = pd.DataFrame(task93_curve_rows)
task93_selected = pd.DataFrame(task93_selected_rows)
print(f"Overall Test forgetting risk = {task93_base_risk:.6f}")
display(task93_selected.round(6))

task93_joint_rows = []
for _, summary_row in task93_summary.iterrows():
    name = summary_row["model"]
    row = {key: summary_row[key] for key in
           ("model", "MAE", "MSE", "ECE10", "AUC")}
    for percent in (10, 20):
        if name == "Train mean":
            risk_k, lift_k = task93_base_risk, 1.0
        else:
            selected = task93_selected.loc[
                (task93_selected["model"] == name)
                & (task93_selected["K_percent"] == percent)
            ].iloc[0]
            risk_k = float(selected["Risk_at_K"])
            lift_k = float(selected["Lift_at_K"])
        row[f"Risk@{percent}%"] = risk_k
        row[f"Lift@{percent}%"] = lift_k
    task93_joint_rows.append(row)
task93_joint = pd.DataFrame(task93_joint_rows)
print("Train mean has no ranking; its Risk/Lift columns show random-tie expectation.")
display(task93_joint.round(6))

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756"]
for name, color in zip(task93_model_names, colors):
    part = task93_curve[task93_curve["model"] == name]
    ax.plot(
        part["K_percent"], part["Lift_at_K"],
        linewidth=2, label=name, color=color,
    )
ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
           label="random expected")
ax.set(
    title="Top-K forgetting-risk lift on Test",
    xlabel="lowest predicted p selected (%)",
    ylabel="Lift@K", xlim=(1, 50),
)
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
plt.show()

task93_delta_names = ["[0, 3) days", "[3, 7) days",
                      "[7, 30) days", "[30, +inf) days"]
task93_delta_id = np.searchsorted(
    np.array([3.0, 7.0, 30.0]), task93_days, side="right"
)
task93_delta_rows = []
for band, band_name in enumerate(task93_delta_names):
    indices = np.flatnonzero(task93_delta_id == band)
    n_band = len(indices)
    base_band_risk = float(task93_risk[indices].mean())
    k = int(np.ceil(n_band * 0.10))
    for name in task93_model_names:
        order = np.argsort(task93_predictions[name][indices], kind="stable")
        selected = indices[order[:k]]
        selected_risk = float(task93_risk[selected].mean())
        task93_delta_rows.append({
            "delta_band": band_name, "model": name, "n_band": n_band,
            "n_selected": k, "band_mean_risk": base_band_risk,
            "Risk@10%": selected_risk,
            "Lift@10%": selected_risk / base_band_risk,
        })
task93_delta_ranking = pd.DataFrame(task93_delta_rows)
display(task93_delta_ranking.round(6))

