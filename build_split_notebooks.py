"""Export the synchronized full notebook back to independent notebooks.

The computational source of truth is pipeline_steps/, not cell positions in
the full notebook. Cell provenance lets this script round-trip all 00--10
sections without restoring superseded analyses or overwriting that shared code.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from build_legacy_notebook import (
    ROOT, TARGET, PROVENANCE_KEY, CELLS, load_notebook, sources_equal,
    write_notebook,
)


def export_notebooks(full):
    manifest = full.get("metadata", {}).get(PROVENANCE_KEY, {})
    if manifest.get("version") != 2:
        raise ValueError("Run python build_legacy_notebook.py before exporting splits.")
    result = {}
    for source in manifest["sources"]:
        name = source["path"]
        if Path(name).name != name or not name.endswith(".ipynb"):
            raise ValueError(f"Invalid source notebook path: {name!r}")
        result[name] = {"cells": [], "metadata": deepcopy(source["metadata"]),
                        "nbformat": source["nbformat"],
                        "nbformat_minor": source["nbformat_minor"]}
    for original in full["cells"]:
        provenance = original.get("metadata", {}).get(PROVENANCE_KEY)
        if provenance is None:
            continue  # Full-notebook introduction.
        if provenance.get("role") == "inline_implementation":
            index = provenance["original_index"]
            if "".join(original["source"]) != CELLS[index]:
                raise ValueError(
                    f"Inline implementation {index} differs from pipeline_steps/. "
                    "Apply the training-code change there and regenerate the full notebook "
                    "before exporting split notebooks."
                )
            continue
        cell = deepcopy(original)
        del cell["metadata"][PROVENANCE_KEY]
        cell["id"] = provenance["cell_id"]
        if provenance.get("bootstrap_replacements"):
            text = "".join(cell["source"])
            for standalone, placeholder in provenance["bootstrap_replacements"]:
                if text.count(placeholder) != 1:
                    raise ValueError(f"Missing bootstrap marker in {original['id']}")
                text = text.replace(placeholder, standalone)
            cell["source"] = text.splitlines(True)
        result[provenance["notebook"]]["cells"].append(cell)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Check source equivalence without changing any files.")
    args = parser.parse_args()
    exported = export_notebooks(load_notebook(TARGET))
    mismatches = []
    for name, notebook in exported.items():
        path = ROOT / name
        existing = load_notebook(path) if path.is_file() else None
        equal = existing is not None and sources_equal(existing, notebook)
        if args.check:
            if not equal:
                mismatches.append(name)
            continue
        if equal:
            continue  # Preserve existing execution outputs and metadata.
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell.update(execution_count=None, outputs=[])
        write_notebook(path, notebook)
        print(f"Updated {name}; outputs cleared because its source changed.")
    if mismatches:
        raise SystemExit("Source differences: " + ", ".join(mismatches))
    print(f"{'Verified' if args.check else 'Synchronized'} {len(exported)} independent notebooks; "
          "pipeline_steps/ was left unchanged.")


if __name__ == "__main__":
    main()
