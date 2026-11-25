"""Build a standalone notebook with the complete 00--10 code in ordinary cells."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from pipeline_cells import CELLS


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "legacy_full_pipeline.ipynb"
PROVENANCE_KEY = "full_pipeline_sync"
# These are the same computations called by Pipeline, in execution order.
# The generated notebook contains the code itself, not imports, exec strings,
# Pipeline wrappers, or dependencies on the generator at runtime.
INLINE_STAGES = {
    "00": (1, 3, 35, 38, 5, 7),
    "01": (),
    "02": (40, 43, 46),
    "03": (48, 50, 52, 54),
    "04": (56, 58, 61, 63),
    "05": (66, 68, 71, 73),
    "06": (76, 78, 80),
    "07": (83,),
    "08": (),
    "09": (),
    "10": (98,),
}


def inline_bootstrap(text):
    """Replace the split notebook's three orchestration lines, reversibly."""
    replacements = [
        ("from experiment_pipeline import Pipeline\n",
         "# All required implementation code is included in this notebook.\n"),
        ("run = Pipeline()\n",
         "# Execute the stages in order in this kernel.\n"),
    ]
    calls = re.findall(r"(?m)^globals\(\)\.update\(run\.\w+\(\)\)\n", text)
    if len(calls) != 1:
        raise ValueError("Expected exactly one Pipeline stage call in each bootstrap.")
    replacements.append((calls[0], "# The inline cells above have completed this stage.\n"))
    for original, replacement in replacements:
        if text.count(original) != 1:
            raise ValueError(f"Unexpected bootstrap: {original!r}")
        text = text.replace(original, replacement)
    return text, replacements


def load_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_notebook(path, notebook):
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def sources_equal(left, right):
    """Compare executable/text content, ignoring execution outputs and UI metadata."""
    return ([(c["cell_type"], "".join(c["source"])) for c in left["cells"]]
            == [(c["cell_type"], "".join(c["source"])) for c in right["cells"]])


def split_paths():
    paths = sorted(p for p in ROOT.glob("*.ipynb")
                   if re.fullmatch(r"(?:0[0-9]|10)_.+\.ipynb", p.name))
    if [p.name[:2] for p in paths] != [f"{i:02}" for i in range(11)]:
        raise ValueError("Expected exactly one notebook for each stage 00 through 10.")
    return paths


def build_full():
    paths = split_paths()
    sources = [(p, load_notebook(p)) for p in paths]
    intro = (
        "# Legacy full pipeline · Standalone 完整项目\n\n"
        "本 notebook 按顺序整合当前全部 11 本拆分版 notebook：数据准备与审计、EDA、"
        "HLR 基线与特征扩展、随机森林、LightGBM、N-HLR、统一比较、非线性消融、"
        "MAE/RMSE 误差分析，以及 RF–LightGBM 相似性诊断。\n\n"
        "数据清洗、特征工程、全部模型的定义、选参、训练、评价和画图代码均已展开为"
        "本文件中的普通代码单元。运行时无需 `experiment_pipeline.py`、`pipeline_cells.py`、"
        "`pipeline_steps/` 或其他 notebook。\n\n"
        "**独立运行：**将本文件放入一个文件夹，准备 Python 3.11 和下方依赖，"
        "把原始文件放在 `data/raw/settles.acl16.learning_traces.13m.csv.gz`，"
        "执行 **Restart Kernel and Run All Cells**。清洗数据、特征缓存、审计报告和图会"
        "自动生成；十二个主模型在本 notebook 中从头训练，不需要预先训练的模型文件。"
        "08 节统一使用 **-lex** 输入与"
        "三个配对种子（42、43、44），09 节误差分解比较 **HLR 与 RF-p**。\n\n"
        "非线性消融支持逐模型缓存续跑；没有缓存时会完成六次训练。"
        "首次运行需要完整数据审计与模型训练，耗时和内存需求与完整拆分版实验相同。\n\n"
        "**安装依赖：**\n\n```text\n"
        + (ROOT / "requirements.txt").read_text(encoding="utf-8").strip()
        + "\n```\n\n将上述包安装到所选 kernel 的 Python 环境即可。\n\n"
        "**内容对应：**按以下拆分版 notebook 的顺序保留所有实验、图表与说明；"
        "主模型的计算代码从其共用实现逐字展开。以下文件只说明来源，运行时无需携带。\n\n"
        + "\n".join(f"- `{p.name}`" for p in paths) + "\n"
    )
    cells = [
        {"cell_type": "markdown", "id": "full-introduction", "metadata": {},
         "source": intro.splitlines(True)},
    ]
    manifest = []
    for path, notebook in sources:
        manifest.append({"path": path.name, "metadata": deepcopy(notebook["metadata"]),
                         "nbformat": notebook["nbformat"],
                         "nbformat_minor": notebook["nbformat_minor"]})
        bootstraps = 0
        for index, original in enumerate(notebook["cells"]):
            cell = deepcopy(original)
            cell_id = cell.get("id", f"cell-{index:03}")
            cell["id"] = f"{path.name[:2]}-{cell_id}"
            provenance = {"notebook": path.name, "cell_id": cell_id}
            if cell["cell_type"] == "code":
                text = "".join(cell["source"])
                if "run = Pipeline()\n" in text:
                    stage = path.name[:2]
                    for original_index in INLINE_STAGES[stage]:
                        cells.append({
                            "cell_type": "code", "id": f"inline-{stage}-{original_index:03}",
                            "metadata": {PROVENANCE_KEY: {
                                "role": "inline_implementation", "stage": stage,
                                "original_index": original_index,
                            }},
                            "source": CELLS[original_index].splitlines(True),
                            "execution_count": None, "outputs": [],
                        })
                    text, replacements = inline_bootstrap(text)
                    provenance["bootstrap_replacements"] = replacements
                    bootstraps += 1
                cell["source"] = text.splitlines(True)
                cell.update(execution_count=None, outputs=[])
            cell["metadata"][PROVENANCE_KEY] = provenance
            cells.append(cell)
        if bootstraps != 1:
            raise ValueError(f"Expected one Pipeline bootstrap in {path.name}; got {bootstraps}.")
    metadata = deepcopy(sources[0][1]["metadata"])
    metadata[PROVENANCE_KEY] = {"version": 2, "standalone": True, "sources": manifest}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Check source equivalence without changing any files.")
    args = parser.parse_args()
    notebook = build_full()
    existing = load_notebook(TARGET) if TARGET.is_file() else None
    if args.check:
        if existing is None or not sources_equal(existing, notebook):
            raise SystemExit("Full notebook is out of sync; run python build_legacy_notebook.py.")
        print("Verified standalone notebook: all 11 notebooks and inline implementations match.")
        return
    # Keep a fully executed notebook intact when regeneration would change nothing.
    if existing is not None and sources_equal(existing, notebook):
        print("Full notebook is already synchronized; existing outputs were preserved.")
        return
    write_notebook(TARGET, notebook)
    print(f"Generated {TARGET.name}: {len(notebook['cells'])} cells from 11 notebooks; "
          "run all cells to produce fresh outputs.")


if __name__ == "__main__":
    main()
