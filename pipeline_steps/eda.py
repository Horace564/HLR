"""Original experiment code, executed by experiment_pipeline.py."""

# === ORIGINAL CELL 010 ===
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SAMPLE_FRAC = 0.02
CHUNK_SIZE = 200_000
SAMPLE_SEED = 42
EDA_COLUMNS = [
    "p_recall", "timestamp", "delta", "history_seen", "history_correct",
    "user_id", "lexeme_id", "learning_language", "ui_language",
]
text_dtypes = {
    "user_id": "string", "lexeme_id": "string",
    "learning_language": "string", "ui_language": "string",
}
parts = []
user_counts = Counter()
lexeme_counts = Counter()
pair_counts = Counter()
pair_recall_sums = Counter()
full_p_zero = full_p_one = 0
full_row_count = 0
chunk_count = 0

for chunk_index, chunk in enumerate(pd.read_csv(
    data_path, compression="gzip", chunksize=CHUNK_SIZE,
    usecols=EDA_COLUMNS, dtype=text_dtypes,
)):
    chunk_count += 1
    full_row_count += len(chunk)
    full_p_zero += int(chunk["p_recall"].eq(0).sum())
    full_p_one += int(chunk["p_recall"].eq(1).sum())

    user_counts.update(chunk["user_id"].value_counts().to_dict())
    lexeme_key = chunk["learning_language"].str.cat(chunk["lexeme_id"], sep=":")
    lexeme_counts.update(lexeme_key.value_counts().to_dict())

    pair_chunk = chunk.groupby(
        ["ui_language", "learning_language"], dropna=False
    )["p_recall"].agg(["size", "sum"])
    for pair, stats in pair_chunk.iterrows():
        pair_counts[pair] += int(stats["size"])
        pair_recall_sums[pair] += float(stats["sum"])

    parts.append(chunk.sample(frac=SAMPLE_FRAC, random_state=SAMPLE_SEED + chunk_index))

eda = pd.concat(parts, ignore_index=True)
eda["delta_days"] = eda["delta"] / 86_400
assert sum(user_counts.values()) == full_row_count
assert sum(lexeme_counts.values()) == full_row_count
assert sum(pair_counts.values()) == full_row_count
print(f"全文件：{full_row_count:,} 行，{chunk_count} 个分块")
print(f"EDA 抽样：{len(eda):,} 行，约 {len(eda)/full_row_count:.2%}；seed={SAMPLE_SEED}")
print(f"全量用户：{len(user_counts):,}；全量语言:词项键：{len(lexeme_counts):,}")
display(eda.head())

# === ORIGINAL CELL 013 ===
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(eda["p_recall"], bins=np.linspace(0, 1, 21), color="#4C78A8", edgecolor="white")
ax.set(xlabel="p_recall", ylabel="抽样行数", title="图 1 · 当前 session 的回忆比例")
ax.set_xlim(0, 1)
plt.tight_layout()
plt.show()
print(f"全量 p_recall=0：{full_p_zero:,}（{full_p_zero/full_row_count:.2%}）")
print(f"全量 p_recall=1：{full_p_one:,}（{full_p_one/full_row_count:.2%}）")
print(f"全量中间值：{full_row_count-full_p_zero-full_p_one:,}（{1-(full_p_zero+full_p_one)/full_row_count:.2%}）")

# === ORIGINAL CELL 016 ===
positive_days = eda.loc[eda["delta_days"] > 0, "delta_days"]
fig, ax = plt.subplots(figsize=(9, 4))
if len(positive_days):
    log_bins = np.logspace(
        np.log10(positive_days.min()), np.log10(positive_days.max()), 55
    )
    ax.hist(positive_days, bins=log_bins, color="#F58518", edgecolor="white")
    ax.set_xscale("log")
ax.set(xlabel="delta（天，对数刻度）", ylabel="抽样行数", title="图 2 · 复习间隔的长尾")
plt.tight_layout()
plt.show()
print("抽样中 delta=0：", int(eda["delta"].eq(0).sum()))
display(eda["delta_days"].quantile([0, .25, .5, .75, .9, .99, 1]).rename("delta_days"))

# === ORIGINAL CELL 019 ===
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, field, color in zip(
    axes, ["history_seen", "history_correct"], ["#54A24B", "#B279A2"]
):
    values = eda[field].to_numpy() + 1
    bins = np.logspace(0, np.log10(values.max() + 1), 50)
    ax.hist(values, bins=bins, color=color, edgecolor="white")
    ax.set_xscale("log")
    ax.set(xlabel=f"{field} + 1（对数刻度）", ylabel="抽样行数", title=field)
plt.suptitle("图 3 · 当前 session 之前的练习历史")
plt.tight_layout()
plt.show()
display(eda[["history_seen", "history_correct"]].quantile([0, .5, .9, .99, 1]))

# === ORIGINAL CELL 022 ===
delta_edges = [-np.inf, 1/24, 1, 7, 30, 90, 365, np.inf]
delta_labels = ["<1 小时", "1 小时–1 天", "1–7 天", "7–30 天",
                "30–90 天", "90–365 天", "≥365 天"]
eda["delta_bin"] = pd.cut(
    eda["delta_days"], bins=delta_edges, labels=delta_labels, right=False
)
delta_summary = eda.groupby("delta_bin", observed=False)["p_recall"].agg(
    mean_recall="mean", n="size"
)
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].bar(delta_labels, delta_summary["mean_recall"], color="#4C78A8")
axes[0].set(ylabel="平均 p_recall", ylim=(0, 1), title="图 4 · 按 delta 分组")
axes[1].bar(delta_labels, delta_summary["n"], color="#9ECAE9")
axes[1].set(ylabel="抽样行数", xlabel="delta 区间")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
display(delta_summary)

# === ORIGINAL CELL 025 ===
seen_edges = [-1, 0, 1, 2, 5, 10, 20, 50, 100, 500, np.inf]
seen_labels = ["0", "1", "2", "3–5", "6–10", "11–20",
               "21–50", "51–100", "101–500", ">500"]
eda["seen_bin"] = pd.cut(
    eda["history_seen"], bins=seen_edges, labels=seen_labels
)
seen_summary = eda.groupby("seen_bin", observed=False)["p_recall"].agg(
    mean_recall="mean", n="size"
)
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].bar(seen_labels, seen_summary["mean_recall"], color="#54A24B")
axes[0].set(ylabel="平均 p_recall", ylim=(0, 1), title="图 5 · 按 history_seen 分组")
axes[1].bar(seen_labels, seen_summary["n"], color="#A1D99B")
axes[1].set(ylabel="抽样行数", xlabel="history_seen 区间")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
display(seen_summary)

# === ORIGINAL CELL 028 ===
pair_summary = pd.DataFrame([
    {
        "ui_language": ui, "learning_language": learning,
        "n": count, "mean_recall": pair_recall_sums[(ui, learning)] / count,
    }
    for (ui, learning), count in pair_counts.items()
]).sort_values("n", ascending=True).reset_index(drop=True)
pair_summary["pair"] = (
    pair_summary["ui_language"] + " → " + pair_summary["learning_language"]
)
positions = np.arange(len(pair_summary))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].barh(positions, pair_summary["n"], color="#72B7B2")
axes[0].set(yticks=positions, yticklabels=pair_summary["pair"],
            xlabel="全量行数", title="样本量")
axes[1].barh(positions, pair_summary["mean_recall"], color="#ECA82C")
axes[1].set(xlabel="平均 p_recall", xlim=(0, 1), title="回忆比例")
fig.suptitle("图 6 · UI 语言 → 学习语言")
plt.tight_layout()
plt.show()
display(pair_summary[["pair", "n", "mean_recall"]])

# === ORIGINAL CELL 031 ===
user_frequency = np.fromiter(user_counts.values(), dtype=np.int64)
lexeme_frequency = np.fromiter(lexeme_counts.values(), dtype=np.int64)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, values, color, label in zip(
    axes,
    [user_frequency, lexeme_frequency],
    ["#4C78A8", "#F58518"],
    ["用户", "语言:词项"],
):
    bins = np.logspace(0, np.log10(values.max() + 1), 45)
    ax.hist(values, bins=bins, color=color, edgecolor="white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel=f"每个{label}的观测行数（对数刻度）",
           ylabel=f"{label}数量（对数刻度）", title=label)
plt.suptitle("图 7 · 实体观测次数的稀疏性")
plt.tight_layout()
plt.show()
print(f"用户：{len(user_frequency):,} 个；≤5 行：{np.mean(user_frequency <= 5):.2%}")
print(f"语言:词项：{len(lexeme_frequency):,} 个；≤5 行：{np.mean(lexeme_frequency <= 5):.2%}")
display(pd.DataFrame({
    "user_rows": pd.Series(user_frequency).quantile([0, .5, .9, .99, 1]),
    "lexeme_rows": pd.Series(lexeme_frequency).quantile([0, .5, .9, .99, 1]),
}))


# === ORIGINAL CELL 105 ===
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

session_eda_rows = Counter()
session_eda_zero = Counter()
session_eda_one = Counter()
session_eda_total = 0
for session_eda_chunk in pd.read_csv(
    data_path, compression="gzip", usecols=["session_seen", "p_recall"], chunksize=200_000,
):
    session_eda_seen = session_eda_chunk["session_seen"]
    session_eda_p = session_eda_chunk["p_recall"]
    if not (session_eda_seen.notna().all()
            and np.isfinite(session_eda_seen.to_numpy()).all()
            and session_eda_seen.gt(0).all()
            and session_eda_seen.eq(np.floor(session_eda_seen)).all()):
        raise ValueError("session_seen 必须是非缺失的正整数；本分析不自动删除异常行。")
    if not session_eda_p.between(0, 1).all():
        raise ValueError("p_recall 必须在 [0, 1] 内且非缺失；本分析不自动删除异常行。")
    session_eda_rows.update(session_eda_seen.value_counts().to_dict())
    session_eda_zero.update(session_eda_seen[session_eda_p.eq(0)].value_counts().to_dict())
    session_eda_one.update(session_eda_seen[session_eda_p.eq(1)].value_counts().to_dict())
    session_eda_total += len(session_eda_chunk)

session_eda_summary = pd.DataFrame([
    {"session_seen": int(k), "rows": int(n),
     "p_zero_rows": int(session_eda_zero[k]), "p_one_rows": int(session_eda_one[k])}
    for k, n in sorted(session_eda_rows.items())
]).set_index("session_seen")
session_eda_summary["p_partial_rows"] = (
    session_eda_summary["rows"] - session_eda_summary["p_zero_rows"]
    - session_eda_summary["p_one_rows"]
)
session_eda_summary["row_share"] = session_eda_summary["rows"] / session_eda_total
for session_eda_kind in ["zero", "partial", "one"]:
    session_eda_summary[f"p_{session_eda_kind}_rate"] = (
        session_eda_summary[f"p_{session_eda_kind}_rows"] / session_eda_summary["rows"]
    )
assert int(session_eda_summary["rows"].sum()) == session_eda_total > 0
assert session_eda_summary["p_partial_rows"].ge(0).all()
np.testing.assert_allclose(
    session_eda_summary[["p_zero_rate", "p_partial_rate", "p_one_rate"]].sum(axis=1), 1.0
)

if "full_row_count" in globals():
    assert session_eda_total == full_row_count
    assert int(session_eda_summary["p_zero_rows"].sum()) == full_p_zero
    assert int(session_eda_summary["p_one_rows"].sum()) == full_p_one
if 1 in session_eda_summary.index:
    assert session_eda_summary.loc[1, "p_partial_rows"] == 0

print(f"全量扫描 {session_eda_total:,} 条去重记录；未抽样、未裁剪标签。")
print(f"session_seen 范围：{session_eda_summary.index.min()}–{session_eda_summary.index.max()}；"
      f"实际出现 {len(session_eda_summary)} 种次数。")
session_eda_distribution_table = session_eda_summary[["rows", "row_share"]].rename(columns={
    "rows": "去重记录数", "row_share": "占全部记录比例",
}).copy()
session_eda_distribution_table["去重记录数"] = session_eda_distribution_table["去重记录数"].map("{:,}".format)
session_eda_distribution_table["占全部记录比例"] = session_eda_distribution_table["占全部记录比例"].map("{:.4%}".format)
display(session_eda_distribution_table)

session_eda_positions = np.arange(len(session_eda_summary))
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.bar(session_eda_positions, session_eda_summary["rows"], color="#4C78A8")
ax.set_yscale("log")
ax.set_xticks(session_eda_positions, session_eda_summary.index.astype(str))
ax.set(xlabel="当前词项在本次练习中的出现次数（session_seen）",
       ylabel="去重记录数（对数刻度）",
       title=f"图 8 · session_seen 分布（全量 {session_eda_total:,} 行）")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# === ORIGINAL CELL 106 ===
session_eda_rate_table = session_eda_summary[[
    "rows", "p_zero_rows", "p_partial_rows", "p_one_rows",
    "p_zero_rate", "p_partial_rate", "p_one_rate",
]].rename(columns={
    "rows": "去重记录数", "p_zero_rows": "全错记录数", "p_partial_rows": "部分正确记录数",
    "p_one_rows": "全对记录数", "p_zero_rate": "p_recall=0 比例",
    "p_partial_rate": "0<p_recall<1 比例", "p_one_rate": "p_recall=1 比例",
})
session_eda_rate_display = session_eda_rate_table.copy()
for session_eda_column in ["去重记录数", "全错记录数", "部分正确记录数", "全对记录数"]:
    session_eda_rate_display[session_eda_column] = session_eda_rate_display[session_eda_column].map("{:,}".format)
for session_eda_column in ["p_recall=0 比例", "0<p_recall<1 比例", "p_recall=1 比例"]:
    session_eda_rate_display[session_eda_column] = session_eda_rate_display[session_eda_column].map("{:.4%}".format)
display(session_eda_rate_display)

fig, ax = plt.subplots(figsize=(12, 5.2))
session_eda_bottom = np.zeros(len(session_eda_summary))
for session_eda_column, session_eda_label, session_eda_color in [
    ("p_zero_rate", "全错（p_recall=0）", "#E45756"),
    ("p_partial_rate", "部分正确（0<p_recall<1）", "#F2CF5B"),
    ("p_one_rate", "全对（p_recall=1）", "#4C78A8"),
]:
    ax.bar(session_eda_positions, session_eda_summary[session_eda_column],
           bottom=session_eda_bottom, label=session_eda_label, color=session_eda_color)
    session_eda_bottom += session_eda_summary[session_eda_column].to_numpy()
ax.set_xticks(session_eda_positions, session_eda_summary.index.astype(str))
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.set(xlabel="当前词项在本次练习中的出现次数（session_seen）",
       ylabel="同一 session_seen 组内的记录比例",
       title="图 9 · 不同 session_seen 下的回忆比例组成（全量精确累计）")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

session_eda_all_one = int(session_eda_summary["p_one_rows"].sum())
if 1 in session_eda_summary.index:
    session_eda_single = session_eda_summary.loc[1]
    print(f"session_seen=1：{int(session_eda_single['rows']):,} 行，"
          f"占全部记录 {session_eda_single['row_share']:.2%}；"
          f"其中全错 {session_eda_single['p_zero_rate']:.2%}，"
          f"全对 {session_eda_single['p_one_rate']:.2%}，部分正确为 0。")
    print(f"全部 p_recall=1 记录中，session_seen=1 占 "
          f"{session_eda_single['p_one_rows'] / session_eda_all_one:.2%}。")
print(f"全量：p_recall=0 占 {sum(session_eda_zero.values()) / session_eda_total:.2%}；"
      f"p_recall=1 占 {session_eda_all_one / session_eda_total:.2%}。")
