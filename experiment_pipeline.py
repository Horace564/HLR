"""Independent notebook entry points for the Duolingo experiment.

The computations are kept verbatim in pipeline_cells.py. This module supplies
fresh-kernel orchestration and versioned, reusable experiment artifacts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from IPython.display import display

from pipeline_cells import CELLS


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "data" / "artifacts"


class Pipeline:
    def __init__(self):
        self.ctx = {"__builtins__": __builtins__, "__name__": "experiment_pipeline",
                    "np": np, "pd": pd, "display": display}
        self.done = set()

    def _run(self, *indices):
        for index in indices:
            exec(compile(CELLS[index], f"pipeline_cells.py:{index}", "exec"), self.ctx)

    def _definitions(self, index, *, constants=False):
        """Load function and class definitions without executing model fits."""
        tree = ast.parse(CELLS[index])
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            elif constants and isinstance(node, ast.Assign):
                if all(isinstance(t, ast.Name) and t.id.startswith("NN_") for t in node.targets):
                    nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]),
                     f"pipeline_cells.py:{index}:definitions", "exec"), self.ctx)

    def _signature(self, stage, indices):
        c = self.ctx
        source = c["source_sha256"]
        versions = {"numpy": np.__version__, "pandas": pd.__version__}
        if stage in {"rf", "lgbm", "comparison"}:
            import sklearn
            versions["sklearn"] = sklearn.__version__
        if stage in {"lgbm", "comparison"}:
            import lightgbm
            versions["lightgbm"] = lightgbm.__version__
        material = {"stage": stage, "source_sha256": source,
                    "n_rows": c["n_rows"], "train_end": c["train_end"],
                    "test_start": c["test_start"], "versions": versions,
                    "cells": {str(i): hashlib.sha256(CELLS[i].encode()).hexdigest()
                              for i in indices}}
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest(), material

    def _artifact(self, stage, indices):
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        signature, material = self._signature(stage, indices)
        return ARTIFACT_DIR / f"{stage}-{signature[:16]}.joblib", material

    def _load(self, stage, indices):
        path, _ = self._artifact(stage, indices)
        if not path.is_file():
            return None
        payload = joblib.load(path)
        print(f"Loaded {stage} results: {path}")
        return payload

    def _save(self, stage, indices, values):
        path, material = self._artifact(stage, indices)
        descriptor = path.with_suffix(".json")
        with tempfile.NamedTemporaryFile(dir=ARTIFACT_DIR, suffix=".partial",
                                         delete=False) as handle:
            partial = Path(handle.name)
        try:
            joblib.dump(values, partial, compress=3)
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
        descriptor.write_text(json.dumps(material, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        print(f"Saved {stage} results: {path}")

    def _store_context(self, stage, indices, names):
        self._save(stage, indices, {name: self.ctx[name] for name in names})

    def _public(self):
        return {key: value for key, value in self.ctx.items() if not key.startswith("__")}

    def prepare(self):
        if "prepare" not in self.done:
            self._run(1, 3, 35, 38)
            self.done.add("prepare")
        return self._public()

    def audit(self):
        self.prepare()
        self._run(5, 7)
        return self._public()

    def eda(self):
        self.prepare()
        return self._public()

    def baselines(self):
        self.prepare()
        if "baselines" in self.done:
            return self._public()
        indices = (35, 40, 43, 46)
        values = self._load("baselines", indices)
        if values is None:
            self._run(40, 43, 46)
            self._store_context("baselines", indices, (
                "hlr_weights", "hlr_results", "hlr_lexless_weights",
                "hlr_lexless_results", "mean_constant_p", "mean_constant_baseline"))
        else:
            self.ctx.update(values)
        self.done.add("baselines")
        return self._public()

    def hlr_experiments(self):
        self.prepare()
        if "hlr_experiments" in self.done:
            return self._public()
        self._definitions(48)
        indices = (35, 38, 48, 50, 52, 54)
        values = self._load("hlr_experiments", indices)
        if values is None:
            self._run(50, 52, 54)
            self._store_context("hlr_experiments", indices, (
                "hlr_user_weights", "hlr_user_layout", "hlr_user_resource",
                "hlr_user_validation", "hlr_user_by_language", "hlr_user_by_user",
                "hlr_lang_weights", "hlr_lang_layout", "hlr_lang_resource",
                "hlr_lang_validation", "hlr_lang_by_language", "hlr_lang_by_user",
                "hlr_both_weights", "hlr_both_layout", "hlr_both_resource",
                "hlr_both_validation", "hlr_both_by_language", "hlr_both_by_user"))
        else:
            self.ctx.update(values)
        self.done.add("hlr_experiments")
        return self._public()

    def _tree_helpers(self):
        if "tree_helpers" in self.done:
            return
        self._definitions(48)
        self._definitions(56)
        self._definitions(58)
        self._definitions(61)
        self.ctx.update({"RF_TUNE_N": 200_000, "RF_FINAL_N": self.ctx["train_end"],
                         "RF_BATCH": 200_000})
        self.done.add("tree_helpers")

    def random_forest(self):
        self.prepare()
        if "rf" in self.done:
            return self._public()
        self._tree_helpers()
        indices = (35, 48, 56, 58, 61, 63)
        values = self._load("rf", indices)
        if values is None:
            self._run(56, 58, 61, 63)
            self._store_context("rf", indices, (
                "rf_e1_model", "rf_e2_model", "rf_candidate_results",
                "rf_p_candidate_results", "rf_e1_validation", "rf_e2_validation",
                "rf_e1_resource", "rf_e2_resource", "rf_e1_fit_report",
                "rf_e2_fit_report"))
        else:
            self.ctx.update(values)
        self.done.add("rf")
        return self._public()

    def lightgbm(self):
        self.prepare()
        if "lgbm" in self.done:
            return self._public()
        self._tree_helpers()
        self._definitions(66)
        self._definitions(71)
        indices = (35, 48, 56, 58, 61, 66, 68, 71, 73)
        values = self._load("lgbm", indices)
        if values is None:
            c = self.ctx
            c["RF_TUNE_N"] = min(c["RF_TUNE_N"], c["train_end"])
            c["rf_tune_pool"] = np.sort(np.random.default_rng(c["SEED"]).choice(
                c["train_end"], min(1_000_000, c["train_end"]), replace=False))
            c["rf_tune_indices"] = np.sort(np.random.default_rng(c["SEED"] + 1).choice(
                c.pop("rf_tune_pool"), c["RF_TUNE_N"], replace=False))
            c["val_start"], c["val_stop"] = c["splits"]["Validation"]
            c["n_val"] = c["val_stop"] - c["val_start"]
            c["p_true"] = c["records"]["p_target"][c["val_start"]:c["val_stop"]]
            c["h_true"] = c["records"]["h_observed"][c["val_start"]:c["val_stop"]]
            c["days"] = c["records"]["delta_days"][c["val_start"]:c["val_stop"]]
            c["interior"] = ((c["h_true"] > c["MIN_H"]) &
                             (c["h_true"] < c["MAX_H"]) & (c["days"] > 0))
            c["n_interior"] = int(c["interior"].sum())
            self._run(66, 68, 71, 73)
            self._store_context("lgbm", indices, (
                "lgbm_e3_model", "lgbm_e4_model", "lgbm_e3_candidate_results",
                "lgbm_e4_candidate_results", "lgbm_e3_validation",
                "lgbm_e4_validation", "lgbm_e3_resource", "lgbm_e4_resource",
                "lgbm_e3_fit_report", "lgbm_e4_fit_report"))
        else:
            self.ctx.update(values)
        self.done.add("lgbm")
        return self._public()

    def neural_hlr(self):
        self.baselines()
        if "nn" in self.done:
            return self._public()
        self._definitions(48)
        self._definitions(76, constants=True)
        indices = (35, 40, 43, 48, 76, 78, 80)
        values = self._load("nn", indices)
        if values is None:
            self._run(78, 80)
            values = {name: self.ctx[name] for name in (
                "nn_hlr_trace", "nn_hlr_resource", "nn_hlr_validation",
                "nn_hlr_fit_report", "nn_hlr_lexless_trace",
                "nn_hlr_lexless_resource", "nn_hlr_lexless_validation",
                "nn_hlr_lexless_fit_report")}
            values["nn_hlr_state"] = self.ctx["nn_hlr_model"].state()
            values["nn_hlr_lexless_state"] = self.ctx["nn_hlr_lexless_model"].state()
            self._save("nn", indices, values)
        else:
            self.ctx.update({key: value for key, value in values.items()
                             if not key.endswith("_state")})
            for stem, weights in (("nn_hlr", "hlr_weights"),
                                  ("nn_hlr_lexless", "hlr_lexless_weights")):
                model = self.ctx["NeuralHLR"](self.ctx[weights])
                model.restore(values[f"{stem}_state"])
                self.ctx[f"{stem}_model"] = model
            self.ctx["nn_hlr_val_p_pred"], self.ctx["nn_hlr_val_h_pred"] = (
                self.ctx["nn_validation_predictions"](self.ctx["nn_hlr_model"]))
            self.ctx["nn_hlr_lexless_val_p_pred"], self.ctx["nn_hlr_lexless_val_h_pred"] = (
                self.ctx["nn_validation_predictions"](self.ctx["nn_hlr_lexless_model"]))
        self.done.add("nn")
        return self._public()

    def comparison(self):
        self.prepare()
        if "comparison" in self.done:
            return self._public()
        indices = (35, 38, 40, 43, 46, 48, 50, 52, 54, 56, 58, 61, 63,
                   66, 68, 71, 73, 76, 78, 80, 83)
        values = self._load("comparison", indices)
        if values is None:
            self.baselines()
            self.hlr_experiments()
            self.random_forest()
            self.lightgbm()
            self.neural_hlr()
            self._run(83)
            self._store_context("comparison", indices, (
                "model_order", "test_predictions", "test_true_p", "test_true_h",
                "test_days", "n_test", "final_summary", "validation_summary",
                "test_summary"))
        else:
            self.ctx.update(values)
        self.done.add("comparison")
        return self._public()
