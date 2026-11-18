"""Rebuild the independent notebooks from the archived full notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "legacy_full_pipeline.ipynb"
NOTEBOOK = json.loads(SOURCE.read_text(encoding="utf-8"))
CELLS = NOTEBOOK["cells"]


def source(index):
    return "".join(CELLS[index]["source"])


def cell(kind, content):
    result = {"cell_type": kind, "metadata": {}, "source": content.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def markdown(index):
    return cell("markdown", source(index))


def code(content):
    return cell("code", content)


def notebook(name, cells):
    for index, item in enumerate(cells):
        item["id"] = f"cell-{index:03d}"
    data = {"cells": cells, "metadata": NOTEBOOK.get("metadata", {}),
            "nbformat": 4, "nbformat_minor": 5}
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")


def bootstrap(method):
    return code(f"from experiment_pipeline import Pipeline\n"
                f"run = Pipeline()\n"
                f"globals().update(run.{method}())\n")


def display_code(*names):
    lines = ["from IPython.display import display"]
    for name in names:
        lines.append(f'print("{name}")')
        lines.append(f"display({name})")
    return code("\n".join(lines) + "\n")


groups = {
    "data.py": [1, 3, 5, 7, 35, 38],
    "eda.py": [10, 13, 16, 19, 22, 25, 28, 31],
    "hlr_baselines.py": [40, 43, 46],
    "hlr_experiments.py": [48, 50, 52, 54],
    "random_forest.py": [56, 58, 61, 63],
    "lightgbm.py": [66, 68, 71, 73],
    "neural_hlr.py": [76, 78, 80],
    "evaluation.py": [83, 86, 88, 90, 94, 98, 99, 102, 104],
}
steps_dir = ROOT / "pipeline_steps"
steps_dir.mkdir(exist_ok=True)
for filename, indices in groups.items():
    content = '"""Original experiment code, executed by experiment_pipeline.py."""\n\n'
    for index in indices:
        content += f"# === ORIGINAL CELL {index:03d} ===\n" + source(index).rstrip() + "\n\n"
    (steps_dir / filename).write_text(content, encoding="utf-8")
(ROOT / "pipeline_cells.py").write_text(
    '"""Load readable experiment sections from pipeline_steps/."""\n'
    'from pathlib import Path\nimport re\n\n'
    'CELLS = {}\n'
    'for path in (Path(__file__).resolve().parent / "pipeline_steps").glob("*.py"):\n'
    '    content = path.read_text(encoding="utf-8")\n'
    '    markers = list(re.finditer(r"(?m)^# === ORIGINAL CELL (\\d{3}) ===\\n", content))\n'
    '    for i, match in enumerate(markers):\n'
    '        end = markers[i + 1].start() if i + 1 < len(markers) else len(content)\n'
    '        CELLS[int(match.group(1))] = content[match.end():end].rstrip() + "\\n"\n',
    encoding="utf-8")

notebook("00_data_preparation.ipynb", [
    cell("markdown", "# 数据准备与审计\n\n从原始 gzip 构建去重数据、固定切分及特征缓存，并运行数据审计。\n"),
    bootstrap("audit"),
])

notebook("01_eda.ipynb", [
    cell("markdown", "# 探索性数据分析\n\n从全新 kernel 运行；输入统一为去重后的数据。\n"),
    bootstrap("eda"),
    *[cell(CELLS[i]["cell_type"], source(i)) for i in range(9, 33)],
])

notebook("02_hlr_baselines.ipynb", [
    cell("markdown", "# HLR 与常数基线\n\n包含完整 HLR、HLR -lex 和 Train mean。重复运行会读取已保存的训练结果。\n"),
    bootstrap("baselines"),
    markdown(39), display_code("hlr_results"),
    markdown(42), display_code("hlr_lexless_results"),
    markdown(45), display_code("mean_constant_baseline"),
])

notebook("03_hlr_experiments.ipynb", [
    cell("markdown", "# HLR 类别特征实验\n\n三个扩展模型分别加入用户、语言或两类特征。\n"),
    bootstrap("hlr_experiments"),
    markdown(49), display_code("hlr_user_validation", "hlr_user_resource"),
    markdown(51), display_code("hlr_lang_validation", "hlr_lang_resource"),
    markdown(53), display_code("hlr_both_validation", "hlr_both_resource"),
])

notebook("04_random_forest.ipynb", [
    cell("markdown", "# 随机森林实验\n\nE1 预测 log2 半衰期，E2 直接预测回忆率。\n"),
    bootstrap("random_forest"),
    markdown(55), display_code("rf_candidate_results", "rf_e1_validation", "rf_e1_resource"),
    markdown(60), display_code("rf_p_candidate_results", "rf_e2_validation", "rf_e2_resource"),
])

notebook("05_lightgbm.ipynb", [
    cell("markdown", "# LightGBM 实验\n\nE3 预测 log2 半衰期，E4 直接预测回忆率。\n"),
    bootstrap("lightgbm"),
    markdown(65), display_code("lgbm_e3_candidate_results", "lgbm_e3_validation", "lgbm_e3_resource"),
    markdown(70), display_code("lgbm_e4_candidate_results", "lgbm_e4_validation", "lgbm_e4_resource"),
])

notebook("06_neural_hlr.ipynb", [
    cell("markdown", "# 神经半衰期模型\n\n两个模型分别包含与不包含 lexeme；所需 HLR 初始化权重会自动读取或训练。\n"),
    bootstrap("neural_hlr"),
    markdown(77), display_code("nn_hlr_trace", "nn_hlr_validation", "nn_hlr_resource"),
    markdown(79), display_code("nn_hlr_lexless_trace", "nn_hlr_lexless_validation", "nn_hlr_lexless_resource"),
])

notebook("07_model_comparison.ipynb", [
    cell("markdown", "# 十二个模型的统一比较\n\n优先读取保存的逐行预测；首次缺少结果时自动补齐依赖模型。\n"),
    bootstrap("comparison"),
    markdown(82), display_code("validation_summary", "test_summary"),
    markdown(92), markdown(93), code(source(94)),
])

notebook("08_error_analysis.ipynb", [
    cell("markdown", "# 模型误差与预测分析\n\n直接使用已保存的 Test 预测；若缺少结果则自动补齐。\n"),
    code("from experiment_pipeline import Pipeline\n"
         "run = Pipeline()\n"
         "run.comparison()\n"
         "run.random_forest()  # 11.1 还需要树模型的原始预测\n"
         "globals().update(run.lightgbm())\n"),
    *[cell(CELLS[i]["cell_type"], source(i)) for i in range(85, 91)],
    *[cell(CELLS[i]["cell_type"], source(i)) for i in range(96, 105)],
])

print("Generated readable pipeline_steps/ code and 9 independent notebooks.")
