"""Load readable experiment sections from pipeline_steps/."""
from pathlib import Path
import re

CELLS = {}
for path in (Path(__file__).resolve().parent / "pipeline_steps").glob("*.py"):
    content = path.read_text(encoding="utf-8")
    markers = list(re.finditer(r"(?m)^# === ORIGINAL CELL (\d{3}) ===\n", content))
    for i, match in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(content)
        CELLS[int(match.group(1))] = content[match.end():end].rstrip() + "\n"


# Presentation text is translated without changing the canonical computation
# blocks used for artifact signatures. Both languages reuse the same artifacts.
ENGLISH_TEXT = {
    "原始文件：": "Raw file:",
    "前 5 行：": "First 5 rows:",
    "切分依据：原始 CSV 文件行顺序；随机打乱仅发生在 Train 内的训练阶段。":
        "Split follows the original CSV row order. Shuffling is confined to Train during fitting.",
    "基础特征：": "Base features:",
    "沿用原论文风格：一次打乱 Train，逐行、逐活跃特征更新。":
        "Original training scheme: shuffle Train once and update each active feature for each row.",
    "已建立新目录所需的特征缓存。": "Created the feature cache.",
    "已复用与当前原始数据、切分一致的特征缓存。":
        "Reused the feature cache matching the current data and split.",
    "缓存：": "Cache:",
    "Train 词项数：": "Train lexemes:",
    "；Validation 未见词项行数：": "; Validation rows with unseen lexemes:",
    "；Test 未见词项行数：": "; Test rows with unseen lexemes:",
    "超参数：": "Hyperparameters:",
    "训练已完成；下表分别是完整 Train、Validation 和 Test 的效果：":
        "Training completed. Metrics below cover the full Train, Validation, and Test splits.",
    "找不到 ": "Cannot find ",
    "；请把数据放到项目的 data/raw/ 下。": "; place the dataset under data/raw/.",
    "；输入特征维度：": "; input dimension: ",
    "实际行数超过已计数的行数": "Input contains more rows than the recorded count",
    "发现 history_seen < history_correct 或负数":
        "Found history_seen < history_correct or a negative count",
    "目标值包含非有限数": "Targets contain nonfinite values",
    "缓存行数 ": "Cache row count ",
    " 与原始文件 ": " differs from the source file count ",
    " 不一致": "",
    " 行；耗时 ": " rows; elapsed ",
    " 秒": " seconds",
    "抽样中 delta=0：": "Sample rows with delta=0:",
    "图 3 · 当前 session 之前的练习历史": "Figure 3: Practice history before the current session",
    "<1 小时": "<1 hour",
    "1 小时–1 天": "1 hour–1 day",
    "1–7 天": "1–7 days",
    "7–30 天": "7–30 days",
    "30–90 天": "30–90 days",
    "90–365 天": "90–365 days",
    "≥365 天": ">=365 days",
    "图 6 · UI 语言 → 学习语言": "Figure 6: Interface language to learning language",
    "图 7 · 实体观测次数的稀疏性": "Figure 7: Sparsity of observations per entity",
    "全文件：": "Full file:",
    " 行，": " rows, ",
    " 个分块": " chunks",
    "EDA 抽样：": "EDA sample:",
    " 行，约 ": " rows, approximately ",
    "全量用户：": "All users:",
    "；全量语言:词项键：": "; all language:lexeme keys:",
    "抽样行数": "Sample rows",
    "图 1 · 当前 session 的回忆比例": "Figure 1: Recall proportion in the current session",
    "全量 p_recall=0：": "All rows with p_recall=0:",
    "全量 p_recall=1：": "All rows with p_recall=1:",
    "全量中间值：": "All rows with interior p_recall:",
    "delta（天，对数刻度）": "delta (days, log scale)",
    "图 2 · 复习间隔的长尾": "Figure 2: Long-tailed review intervals",
    "平均 p_recall": "Mean p_recall",
    "图 4 · 按 delta 分组": "Figure 4: Groups by review interval",
    "delta 区间": "delta interval",
    "图 5 · 按 history_seen 分组": "Figure 5: Groups by history_seen",
    "history_seen 区间": "history_seen interval",
    "全量行数": "All rows",
    "样本量": "Sample size",
    "回忆比例": "Recall proportion",
    "用户": "user",
    "语言:词项": "language:lexeme",
    "用户：": "Users:",
    " 个；≤5 行：": " entities; <=5 observations:",
    "语言:词项：": "Language:lexeme keys:",
    " + 1（对数刻度）": " + 1 (log scale)",
    "每个": "Each ",
    "的观测行数（对数刻度）": " observation count (log scale)",
    "数量（对数刻度）": "Count (log scale)",
    "[0, 3) 天": "[0, 3) days",
    "[3, 7) 天": "[3, 7) days",
    "[7, 30) 天": "[7, 30) days",
    "[30, +∞) 天": "[30, +∞) days",
    "Test 各 delta 区间的样本数；十二个模型共用这些行。":
        "Test counts by delta interval. All twelve models use the same rows.",
    "：逐行比较十二个模型；逐列比较同一复习间隔。":
        ": compare models by row and review intervals by column.",
    "# 原有 src/data.py 中的基础有效性与完整重复记录审计代码。":
        "# Basic validity and exact-duplicate audits from the original data module.",
    "# 原有 src/audit_history_counts.py 中的全量历史计数审计代码。":
        "# Full historical-count audit from the original audit module.",
    "# 计数与缓存校验都直接读原始 gzip；不依赖旧目录中的报告、词表或模型文件。":
        "# Count rows and validate caches directly against the source gzip.",
    "# 只能从 Train 学到词项列号；从第 3 列开始。":
        "# Learn lexeme columns from Train only, starting at column 3.",
    "# 第 8 节的十二个模型，顺序保持一致":
        "# Keep the twelve-model order from section 8.",
    "# 只横向错开绘图位置；每个点的 y 值仍是表中的真实指标。":
        "# Offset x positions for clarity; retain the actual metric values on y.",
    "# 两种模型在训练前都必须精确复现原始 HLR -lex。":
        "# Both models must reproduce the original HLR -lex before training.",
}


def english_text(text):
    """Translate project-owned text, including fragments inside f-strings."""
    for original in sorted(ENGLISH_TEXT, key=len, reverse=True):
        text = text.replace(original, ENGLISH_TEXT[original])
    return text


def localized_cells(language="en"):
    """Return executable blocks while retaining CELLS for stable cache keys."""
    if language == "en":
        return {index: english_text(source) for index, source in CELLS.items()}
    if language == "zh-CN":
        return dict(CELLS)
    raise ValueError("language must be 'en' or 'zh-CN'")
