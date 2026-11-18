"""Original experiment code, executed by experiment_pipeline.py."""

# === ORIGINAL CELL 001 ===
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

filename = "settles.acl16.learning_traces.13m.csv.gz"
candidates = [Path("data/raw") / filename, Path("../data/raw") / filename]
data_path = next((p.resolve() for p in candidates if p.is_file()), None)
if data_path is None:
    raise FileNotFoundError(f"找不到 {filename}；请把数据放到项目的 data/raw/ 下。")

preview = pd.read_csv(data_path, compression="gzip", nrows=5)
print("原始文件：", data_path)
print("前 5 行：")
display(preview)

# === ORIGINAL CELL 003 ===
import csv
import gzip
import hashlib
import os
import tempfile
from contextlib import ExitStack
import json

CP2_FIELDS = (
    "p_recall", "timestamp", "delta", "user_id", "learning_language",
    "ui_language", "lexeme_id", "lexeme_string", "history_seen",
    "history_correct", "session_seen", "session_correct",
)
raw_candidates = [Path("data/raw") / filename, Path("../data/raw") / filename]
raw_data_path = next((path.resolve() for path in raw_candidates if path.is_file()), None)
if raw_data_path is None:
    raise FileNotFoundError(f"Missing raw data: {filename}")
project_data_dir = raw_data_path.parent.parent
processed_dir = project_data_dir / "processed"
interim_dir = project_data_dir / "interim"
reports_dir = project_data_dir.parent / "reports"
for directory in (processed_dir, interim_dir, reports_dir):
    directory.mkdir(parents=True, exist_ok=True)
cleaned_data_path = processed_dir / "settles.acl16.learning_traces.13m.dedup_v1.csv.gz"
clean_manifest_path = processed_dir / "settles.acl16.learning_traces.13m.dedup_v1.json"
duplicate_report_path = reports_dir / "exact_duplicates_v1.json"

def cp2_file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def cp2_raw_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader)) != CP2_FIELDS:
            raise ValueError("Unexpected raw CSV header")
        yield from enumerate(reader)

def cp2_row_digest(row):
    return hashlib.blake2b(repr(tuple(row)).encode("utf-8"), digest_size=16).digest()

def cp2_find_exact_duplicates(path):
    with tempfile.TemporaryDirectory(prefix="cp2_dedup_", dir=interim_dir) as temp:
        bucket_paths = [Path(temp) / f"{i:02x}.bin" for i in range(256)]
        with ExitStack() as stack:
            buckets = [
                stack.enter_context(bucket.open("wb")) for bucket in bucket_paths
            ]
            raw_rows = 0
            for index, row in cp2_raw_rows(path):
                digest = cp2_row_digest(row)
                buckets[digest[0]].write(digest)
                raw_rows += 1
        candidates = set()
        for bucket_path in bucket_paths:
            seen = set()
            with bucket_path.open("rb") as stream:
                for digest in iter(lambda: stream.read(16), b""):
                    if digest in seen:
                        candidates.add(digest)
                    else:
                        seen.add(digest)
    first_rows = {}
    duplicate_indices = []
    examples = []
    if candidates:
        for index, row in cp2_raw_rows(path):
            digest = cp2_row_digest(row)
            if digest not in candidates:
                continue
            complete_row = tuple(row)
            seen = first_rows.setdefault(digest, [])
            first = next(
                (prior_index for previous, prior_index in seen
                 if previous == complete_row),
                None,
            )
            if first is None:
                seen.append((complete_row, index))
            else:
                duplicate_indices.append(index)
                if len(examples) < 10:
                    examples.append({
                        "first_row_index": first,
                        "duplicate_row_index": index,
                    })
    return raw_rows, duplicate_indices, examples

raw_stat = raw_data_path.stat()
clean_manifest = (
    json.loads(clean_manifest_path.read_text(encoding="utf-8"))
    if clean_manifest_path.is_file() else {}
)
clean_valid = (
    cleaned_data_path.is_file()
    and duplicate_report_path.is_file()
    and clean_manifest.get("version") == 1
    and clean_manifest.get("raw_file") == raw_data_path.name
    and clean_manifest.get("raw_size_bytes") == raw_stat.st_size
    and clean_manifest.get("raw_mtime_ns") == raw_stat.st_mtime_ns
    and clean_manifest.get("clean_size_bytes") == cleaned_data_path.stat().st_size
    and clean_manifest.get("clean_mtime_ns") == cleaned_data_path.stat().st_mtime_ns
    and clean_manifest.get("raw_rows", 0) > 0
    and clean_manifest.get("clean_rows", 0) > 0
)
if not clean_valid:
    raw_rows, duplicate_indices, examples = cp2_find_exact_duplicates(raw_data_path)
    duplicate_set = set(duplicate_indices)
    partial_path = cleaned_data_path.with_suffix(".partial")
    kept = 0
    with gzip.open(partial_path, "wt", encoding="utf-8", newline="",
                   compresslevel=6) as output:
        writer = csv.writer(output)
        writer.writerow(CP2_FIELDS)
        for index, row in cp2_raw_rows(raw_data_path):
            if index not in duplicate_set:
                writer.writerow(row)
                kept += 1
    assert kept == raw_rows - len(duplicate_indices)
    os.replace(partial_path, cleaned_data_path)
    clean_manifest = {
        "version": 1,
        "rule": "exact full CSV row; keep first raw occurrence",
        "raw_file": raw_data_path.name,
        "raw_size_bytes": raw_stat.st_size,
        "raw_mtime_ns": raw_stat.st_mtime_ns,
        "raw_sha256": cp2_file_sha256(raw_data_path),
        "raw_rows": raw_rows,
        "duplicate_rows_after_first": len(duplicate_indices),
        "duplicate_raw_row_indices": duplicate_indices,
        "clean_rows": kept,
        "clean_size_bytes": cleaned_data_path.stat().st_size,
        "clean_mtime_ns": cleaned_data_path.stat().st_mtime_ns,
        "clean_sha256": cp2_file_sha256(cleaned_data_path),
    }
    clean_manifest_path.write_text(
        json.dumps(clean_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    duplicate_report_path.write_text(
        json.dumps({
            "raw_sha256": clean_manifest["raw_sha256"],
            "raw_rows": raw_rows,
            "duplicate_rows_after_first": len(duplicate_indices),
            "duplicate_examples": examples,
            "policy": clean_manifest["rule"],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Completed full raw-data exact-duplicate check and cleaning.")
else:
    print("Reused saved duplicate check and cleaned dataset.")

data_path = cleaned_data_path
assert clean_manifest["clean_rows"] == (
    clean_manifest["raw_rows"] - clean_manifest["duplicate_rows_after_first"]
)
display(pd.DataFrame([{
    "raw_rows": clean_manifest["raw_rows"],
    "exact_duplicates_removed": clean_manifest["duplicate_rows_after_first"],
    "clean_rows": clean_manifest["clean_rows"],
}]))
print("Model and EDA input:", data_path)

# === ORIGINAL CELL 005 ===
# 原有 src/data.py 中的基础有效性与完整重复记录审计代码。
import csv
import gzip
import hashlib
import heapq
import math
import tempfile
from collections import Counter
from contextlib import ExitStack
from typing import Iterator

FIELDS = (
    "p_recall", "timestamp", "delta", "user_id", "learning_language",
    "ui_language", "lexeme_id", "lexeme_string", "history_seen",
    "history_correct", "session_seen", "session_correct",
)


INTEGER_FIELDS = (
    "timestamp", "delta", "history_seen", "history_correct",
    "session_seen", "session_correct",
)


NUMERIC_FIELDS = ("p_recall",) + INTEGER_FIELDS


RATIO_ABS_TOL = 1e-8


READ_BUFFER = 8 * 1024 * 1024


VALIDITY_RULES = (
    ["malformed_csv_row"]
    + [f"missing:{field}" for field in FIELDS]
    + [f"invalid_type:{field}" for field in NUMERIC_FIELDS]
    + [
        "p_recall_out_of_range", "delta_negative", "history_seen_negative",
        "history_correct_negative", "history_correct_gt_seen",
        "session_seen_nonpositive", "session_correct_negative",
        "session_correct_gt_seen", "p_recall_ratio_mismatch",
    ]
)


DIAGNOSTIC_COUNTS = (
    "p_recall_zero", "p_recall_one", "delta_zero", "delta_gt_30_days",
    "delta_gt_365_days", "timestamp_nonpositive",
    "timestamp_backwards_adjacent",
)




def inspect_row(row: dict[str, str | None]) -> tuple[list[str], dict[str, int | float]]:
    """Return all applicable exclusion reasons and successfully parsed numbers.

    Rules are independent, so a row may have more than one reason. A zero or
    one recall target is valid. No session outcome is used as a predictor.
    """
    reasons: list[str] = []
    values: dict[str, int | float] = {}
    if "_extra" in row:
        reasons.append("malformed_csv_row")

    for field in FIELDS:
        raw = row.get(field)
        if raw is None or not raw.strip():
            reasons.append(f"missing:{field}")
            continue
        if field not in NUMERIC_FIELDS:
            continue
        try:
            value = float(raw) if field == "p_recall" else int(raw)
        except (ValueError, TypeError):
            reasons.append(f"invalid_type:{field}")
            continue
        if isinstance(value, float) and not math.isfinite(value):
            reasons.append(f"invalid_type:{field}")
            continue
        values[field] = value

    p = values.get("p_recall")
    delta = values.get("delta")
    h_seen = values.get("history_seen")
    h_correct = values.get("history_correct")
    s_seen = values.get("session_seen")
    s_correct = values.get("session_correct")

    if p is not None and not 0 <= p <= 1:
        reasons.append("p_recall_out_of_range")
    if delta is not None and delta < 0:
        reasons.append("delta_negative")
    if h_seen is not None and h_seen < 0:
        reasons.append("history_seen_negative")
    if h_correct is not None and h_correct < 0:
        reasons.append("history_correct_negative")
    if h_seen is not None and h_correct is not None and h_correct > h_seen:
        reasons.append("history_correct_gt_seen")
    if s_seen is not None and s_seen <= 0:
        reasons.append("session_seen_nonpositive")
    if s_correct is not None and s_correct < 0:
        reasons.append("session_correct_negative")
    if s_seen is not None and s_correct is not None and s_correct > s_seen:
        reasons.append("session_correct_gt_seen")
    if (
        p is not None and 0 <= p <= 1
        and s_seen is not None and s_seen > 0
        and s_correct is not None and 0 <= s_correct <= s_seen
        and not math.isclose(p, s_correct / s_seen, rel_tol=0, abs_tol=RATIO_ABS_TOL)
    ):
        reasons.append("p_recall_ratio_mismatch")
    return reasons, values

quality_report_path = reports_dir / "data_quality_clean_v1.json"
quality_summary = (
    json.loads(quality_report_path.read_text(encoding="utf-8"))
    if quality_report_path.is_file() else {}
)
quality_valid = (
    quality_summary.get("version") == 1
    and quality_summary.get("input_sha256") == clean_manifest["clean_sha256"]
    and quality_summary.get("total_rows") == clean_manifest["clean_rows"]
    and quality_summary.get("scope") == "full_file"
)
if not quality_valid:
    rule_hits = Counter()
    examples = []
    invalid_rows = 0
    total_rows = 0
    with gzip.open(data_path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, restkey="_extra", restval=None)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected cleaned CSV header")
        for index, row in enumerate(reader):
            reasons, _ = inspect_row(row)
            total_rows += 1
            if reasons:
                invalid_rows += 1
                rule_hits.update(reasons)
                if len(examples) < 20:
                    examples.append({"row_index": index, "reasons": reasons})
    if total_rows != clean_manifest["clean_rows"]:
        raise ValueError("Cleaned row count differs from its manifest")
    quality_summary = {
        "version": 1,
        "scope": "full_file",
        "input_sha256": clean_manifest["clean_sha256"],
        "total_rows": total_rows,
        "rows_with_any_basic_issue": invalid_rows,
        "ratio_abs_tolerance": RATIO_ABS_TOL,
        "rule_hit_counts": {
            rule: int(rule_hits[rule]) for rule in VALIDITY_RULES
        },
        "examples": examples,
        "policy": "audit only; stop if any basic validity issue remains",
    }
    quality_report_path.write_text(
        json.dumps(quality_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Completed full cleaned-data basic-validity audit.")
else:
    print("Reused saved basic-validity audit.")

rule_hits = quality_summary["rule_hit_counts"]
validity_groups = [
    ("missing numeric", [key for key in rule_hits
                         if key.startswith("missing:") and
                         key.split(":", 1)[1] in NUMERIC_FIELDS]),
    ("missing categorical/text", [key for key in rule_hits
                                  if key.startswith("missing:") and
                                  key.split(":", 1)[1] not in NUMERIC_FIELDS]),
    ("invalid numeric type", [key for key in rule_hits
                              if key.startswith("invalid_type:")]),
    ("numeric range", [
        "p_recall_out_of_range", "delta_negative",
        "history_seen_negative", "history_correct_negative",
        "session_seen_nonpositive", "session_correct_negative",
    ]),
    ("count relationship", [
        "history_correct_gt_seen", "session_correct_gt_seen",
    ]),
    ("p_recall/session ratio", ["p_recall_ratio_mismatch"]),
    ("malformed CSV", ["malformed_csv_row"]),
]
assert {key for _, keys in validity_groups for key in keys} == set(rule_hits)
display(pd.DataFrame([
    {
        "check": group,
        "rule_hits": sum(rule_hits[key] for key in keys),
        "clean_rows": quality_summary["total_rows"],
    }
    for group, keys in validity_groups
]))
if quality_summary["rows_with_any_basic_issue"]:
    raise ValueError(
        "Basic validity issues remain in the cleaned dataset; "
        "only exact duplicates are removed automatically."
    )

# === ORIGINAL CELL 007 ===
# 原有 src/audit_history_counts.py 中的全量历史计数审计代码。
import itertools

BUCKETS = 256


HISTORY_FIELDS = (
    "user_id", "learning_language", "lexeme_id", "timestamp",
    "history_seen", "history_correct", "session_seen", "session_correct",
)


HISTORY_OUTPUT_FIELDS = (
    "raw_row_index", "user_id", "learning_language", "lexeme_id", "timestamp",
    "history_seen", "history_correct", "session_seen", "session_correct",
    "seen_rollback", "correct_rollback", "seen_cumulative_shortfall",
    "correct_cumulative_shortfall", "prior_max_seen", "prior_max_correct",
    "prior_max_seen_plus_session", "prior_max_correct_plus_session",
)


def bucket_id(key: tuple[str, str, str]) -> int:
    return hashlib.blake2b(repr(key).encode("utf-8"), digest_size=1).digest()[0]


def cp2_audit_history(source: Path, output: Path, *, max_rows: int | None = None) -> dict:
    source = source.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in (
        "rows_scanned", "groups", "timestamp_groups", "same_timestamp_groups_with_different_counts",
        "rows_with_any_anomaly", "rows_with_seen_rollback", "rows_with_correct_rollback",
        "rows_with_seen_cumulative_shortfall", "rows_with_correct_cumulative_shortfall",
        "rows_with_cumulative_shortfall_without_rollback", "groups_with_any_anomaly",
    )}
    examples: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="hlr_count_audit_", dir=interim_dir) as temporary:
        paths = [Path(temporary) / f"{i:02x}.csv" for i in range(BUCKETS)]
        with ExitStack() as stack:
            writers = [csv.writer(stack.enter_context(p.open("w", encoding="utf-8", newline="")))
                       for p in paths]
            with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not set(HISTORY_FIELDS).issubset(reader.fieldnames or []):
                    raise ValueError("Required history fields are missing")
                for index, row in enumerate(reader):
                    if max_rows is not None and index >= max_rows:
                        break
                    key = tuple(row[field] for field in HISTORY_FIELDS[:3])
                    writers[bucket_id(key)].writerow((index, *(row[field] for field in HISTORY_FIELDS)))
                    counts["rows_scanned"] += 1

        with gzip.open(output, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_OUTPUT_FIELDS)
            writer.writeheader()
            for path in paths:
                with path.open("r", encoding="utf-8", newline="") as stream_in:
                    records = [
                        (int(r[0]), r[1], r[2], r[3], int(r[4]), int(r[5]),
                         int(r[6]), int(r[7]), int(r[8]))
                        for r in csv.reader(stream_in)
                    ]
                records.sort(key=lambda r: (r[1], r[2], r[3], r[4], r[0]))
                for _, pair_iter in itertools.groupby(records, key=lambda r: r[1:4]):
                    counts["groups"] += 1
                    prior_seen = prior_correct = None
                    prior_seen_after = prior_correct_after = None
                    group_has_anomaly = False
                    for _, time_iter in itertools.groupby(pair_iter, key=lambda r: r[4]):
                        same_time = list(time_iter)
                        counts["timestamp_groups"] += 1
                        if len({(r[5], r[6], r[7], r[8]) for r in same_time}) > 1:
                            counts["same_timestamp_groups_with_different_counts"] += 1
                        for r in same_time:
                            seen_rollback = prior_seen is not None and r[5] < prior_seen
                            correct_rollback = prior_correct is not None and r[6] < prior_correct
                            seen_short = prior_seen_after is not None and r[5] < prior_seen_after
                            correct_short = prior_correct_after is not None and r[6] < prior_correct_after
                            flags = (seen_rollback, correct_rollback, seen_short, correct_short)
                            if not any(flags):
                                continue
                            group_has_anomaly = True
                            counts["rows_with_any_anomaly"] += 1
                            for flag, name in zip(flags, (
                                "rows_with_seen_rollback", "rows_with_correct_rollback",
                                "rows_with_seen_cumulative_shortfall",
                                "rows_with_correct_cumulative_shortfall",
                            )):
                                counts[name] += int(flag)
                            if (seen_short or correct_short) and not (seen_rollback or correct_rollback):
                                counts["rows_with_cumulative_shortfall_without_rollback"] += 1
                            record = dict(zip(HISTORY_OUTPUT_FIELDS, (
                                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
                                *flags, prior_seen, prior_correct,
                                prior_seen_after, prior_correct_after,
                            )))
                            writer.writerow(record)
                            if len(examples) < 20:
                                examples.append(record)
                        prior_seen = max(prior_seen or 0, *(r[5] for r in same_time))
                        prior_correct = max(prior_correct or 0, *(r[6] for r in same_time))
                        prior_seen_after = max(prior_seen_after or 0, *(r[5] + r[7] for r in same_time))
                        prior_correct_after = max(prior_correct_after or 0, *(r[6] + r[8] for r in same_time))
                    counts["groups_with_any_anomaly"] += int(group_has_anomaly)

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "source": str(source), "source_sha256": digest.hexdigest(),
        "scope": "full_file" if max_rows is None else f"first_{max_rows}_rows",
        "group_key": list(HISTORY_FIELDS[:3]),
        "rule": "Strictly later timestamps only; rollback compares prior maximum history count; cumulative shortfall compares prior maximum history + session count.",
        "same_timestamp_policy": "No ordering or anomaly comparison within a timestamp.",
        "anomaly_rows_file": str(output.resolve()), **counts, "examples": examples,
    }


history_report_path = reports_dir / "history_count_audit_clean_v1.json"
history_anomaly_path = interim_dir / "history_count_anomalies_clean_v1.csv.gz"
history_summary = (
    json.loads(history_report_path.read_text(encoding="utf-8"))
    if history_report_path.is_file() else {}
)
history_valid = (
    history_anomaly_path.is_file()
    and history_summary.get("version") == 1
    and history_summary.get("scope") == "full_file"
    and history_summary.get("source_sha256") == clean_manifest["clean_sha256"]
    and history_summary.get("rows_scanned") == clean_manifest["clean_rows"]
)
if not history_valid:
    history_summary = cp2_audit_history(data_path, history_anomaly_path)
    if history_summary["rows_scanned"] != clean_manifest["clean_rows"]:
        raise ValueError("History audit row count differs from cleaned dataset")
    if history_summary["source_sha256"] != clean_manifest["clean_sha256"]:
        raise ValueError("History audit source hash differs from cleaned dataset")
    history_summary["version"] = 1
    history_summary["policy"] = "report only; retain all history anomalies"
    history_report_path.write_text(
        json.dumps(history_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Completed full cleaned-data history-consistency audit.")
else:
    print("Reused saved history-consistency audit.")

any_history = int(history_summary["rows_with_any_anomaly"])
shortfall_only = int(
    history_summary["rows_with_cumulative_shortfall_without_rollback"]
)
any_rollback = any_history - shortfall_only
display(pd.DataFrame([
    {
        "check": "history rollback",
        "rows": any_rollback,
        "clean_rows": history_summary["rows_scanned"],
    },
    {
        "check": "cumulative shortfall (includes rollback)",
        "rows": any_history,
        "clean_rows": history_summary["rows_scanned"],
    },
    {
        "check": "same timestamp with differing counts",
        "rows": history_summary["same_timestamp_groups_with_different_counts"],
        "clean_rows": history_summary["rows_scanned"],
    },
]))
print("History anomalies are reported and retained in the cleaned dataset.")

# === ORIGINAL CELL 035 ===
import hashlib
import json
import math
import os
import time

# 计数与缓存校验都直接读原始 gzip；不依赖旧目录中的报告、词表或模型文件。
source_sha256 = hashlib.sha256()
with data_path.open("rb") as source:
    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
        source_sha256.update(block)
source_sha256 = source_sha256.hexdigest()

n_rows = sum(len(chunk) for chunk in pd.read_csv(
    data_path, compression="gzip", usecols=["p_recall"], chunksize=500_000
))
train_end = int(0.72 * n_rows)
test_start = int(0.90 * n_rows)
splits = {
    "Train": (0, train_end),
    "Validation": (train_end, test_start),
    "Test": (test_start, n_rows),
}
display(pd.DataFrame([
    {"split": name, "start": start, "stop_exclusive": stop, "rows": stop - start}
    for name, (start, stop) in splits.items()
]))
print("切分依据：原始 CSV 文件行顺序；随机打乱仅发生在 Train 内的训练阶段。")

MIN_H, MAX_H = 15.0 / 1440.0, 274.0
MIN_P, MAX_P = 0.0001, 0.9999
ALPHA, L2, LEARNING_RATE, SEED, EPOCHS = 0.01, 0.1, 0.001, 42, 1
LOG_MIN_H, LOG_MAX_H = math.log2(MIN_H), math.log2(MAX_H)
BASE_FEATURES = [
    "sqrt(1+history_correct)",
    "sqrt(1+history_seen-history_correct)",
    "bias",
]
RECORD_DTYPE = np.dtype([
    ("x_correct", "<f8"), ("x_wrong", "<f8"), ("lexeme_column", "<i4"),
    ("p_target", "<f8"), ("delta_days", "<f8"), ("h_observed", "<f8"),
])

cache_dir = data_path.parent.parent / "interim"
cache_dir.mkdir(parents=True, exist_ok=True)
cache_path = cache_dir / "takeaway_hlr_features_v1.dat"
manifest_path = cache_dir / "takeaway_hlr_features_v1.json"
cache_identity = {
    "version": 1,
    "source_sha256": source_sha256,
    "n_rows": n_rows,
    "train_end": train_end,
    "dtype": repr(RECORD_DTYPE.descr),
}
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
cache_valid = (
    cache_path.is_file()
    and all(manifest.get(key) == value for key, value in cache_identity.items())
    and cache_path.stat().st_size == n_rows * RECORD_DTYPE.itemsize
)

if not cache_valid:
    vocabulary = {}  # 只能从 Train 学到词项列号；从第 3 列开始。
    unseen_validation = unseen_test = 0
    partial_path = cache_path.with_suffix(".partial")
    records_out = np.memmap(partial_path, dtype=RECORD_DTYPE, mode="w+", shape=(n_rows,))
    written = 0
    needed = [
        "p_recall", "delta", "history_seen", "history_correct",
        "learning_language", "lexeme_string",
    ]
    for chunk in pd.read_csv(data_path, compression="gzip", usecols=needed,
                             chunksize=200_000):
        start, stop = written, written + len(chunk)
        if stop > n_rows:
            raise ValueError("实际行数超过已计数的行数")
        keys = chunk["learning_language"].astype(str).str.cat(
            chunk["lexeme_string"].astype(str), sep=":"
        )
        train_count = max(0, min(stop, train_end) - start)
        for key in pd.unique(keys.iloc[:train_count]):
            vocabulary.setdefault(key, 3 + len(vocabulary))
        columns = keys.map(vocabulary).fillna(-1).to_numpy(dtype=np.int32)

        correct = chunk["history_correct"].to_numpy(dtype=np.float64)
        wrong = chunk["history_seen"].to_numpy(dtype=np.float64) - correct
        if np.any(correct < 0) or np.any(wrong < 0):
            raise ValueError("发现 history_seen < history_correct 或负数")
        p = np.clip(chunk["p_recall"].to_numpy(dtype=np.float64), MIN_P, MAX_P)
        days = chunk["delta"].to_numpy(dtype=np.float64) / 86_400.0
        h = np.clip(-days / np.log2(p), MIN_H, MAX_H)
        if not (np.isfinite(p).all() and np.isfinite(h).all()):
            raise ValueError("目标值包含非有限数")

        records_out["x_correct"][start:stop] = np.sqrt(1.0 + correct)
        records_out["x_wrong"][start:stop] = np.sqrt(1.0 + wrong)
        records_out["lexeme_column"][start:stop] = columns
        records_out["p_target"][start:stop] = p
        records_out["delta_days"][start:stop] = days
        records_out["h_observed"][start:stop] = h

        val_left, val_right = max(start, train_end), min(stop, test_start)
        if val_left < val_right:
            unseen_validation += int(np.sum(columns[val_left-start:val_right-start] < 0))
        test_left = max(start, test_start)
        if test_left < stop:
            unseen_test += int(np.sum(columns[test_left-start:] < 0))
        written = stop

    if written != n_rows:
        raise ValueError(f"缓存行数 {written:,} 与原始文件 {n_rows:,} 不一致")
    records_out.flush()
    del records_out
    os.replace(partial_path, cache_path)
    manifest = {
        **cache_identity,
        "vocab_size": len(vocabulary),
        "unseen_validation": unseen_validation,
        "unseen_test": unseen_test,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print("已建立新目录所需的特征缓存。")
else:
    print("已复用与当前原始数据、切分一致的特征缓存。")

records = np.memmap(cache_path, dtype=RECORD_DTYPE, mode="r", shape=(n_rows,))
vocab_size = int(manifest["vocab_size"])
print(f"缓存：{cache_path}；{cache_path.stat().st_size / 2**20:.1f} MiB")
print("基础特征：", BASE_FEATURES)
print(f"Train 词项数：{vocab_size:,}；Validation 未见词项行数："
      f"{manifest['unseen_validation']:,}；Test 未见词项行数：{manifest['unseen_test']:,}")

def train_hlr(dimension):
    """沿用原论文风格：一次打乱 Train，逐行、逐活跃特征更新。"""
    weights = [0.0] * dimension
    feature_counts = [0] * dimension
    order = np.arange(train_end, dtype=np.int32)
    np.random.default_rng(SEED).shuffle(order)
    started = time.perf_counter()
    x0_all, x1_all = records["x_correct"], records["x_wrong"]
    lex_all, p_all = records["lexeme_column"], records["p_target"]
    days_all, h_all = records["delta_days"], records["h_observed"]
    for step, row_index in enumerate(order, start=1):
        x0, x1 = float(x0_all[row_index]), float(x1_all[row_index])
        column = int(lex_all[row_index])
        p, days, h_true = float(p_all[row_index]), float(days_all[row_index]), float(h_all[row_index])
        score = weights[0] * x0 + weights[1] * x1 + weights[2]
        if dimension > 3 and 3 <= column < dimension:
            score += weights[column]
        h_pred = math.exp2(min(max(score, LOG_MIN_H), LOG_MAX_H))
        p_pred = min(max(math.exp2(-days / h_pred), MIN_P), MAX_P)
        grad_p = 2.0 * (p_pred - p) * math.log(2.0)**2 * p_pred * (days / h_pred)
        grad_h = 2.0 * (h_pred - h_true) * math.log(2.0) * h_pred
        active = ((0, x0), (1, x1), (2, 1.0))
        if dimension > 3 and 3 <= column < dimension:
            active += ((column, 1.0),)
        for k, value in active:
            rate = LEARNING_RATE / ((1.0 + p) * math.sqrt(1 + feature_counts[k]))
            weights[k] -= rate * grad_p * value
            weights[k] -= rate * ALPHA * grad_h * value
            weights[k] -= rate * L2 * weights[k]
            feature_counts[k] += 1
        if step % 1_000_000 == 0 or step == train_end:
            print(f"  {step:,}/{train_end:,} 行；耗时 {time.perf_counter()-started:.1f} 秒")
    return np.asarray(weights, dtype=np.float64)

def auc_fixed(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def spearman_with_ties(observed, predicted):
    x_rank = pd.Series(observed).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(predicted).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])

def evaluate_hlr(weights, split_name, batch_size=200_000):
    start, stop = splits[split_name]
    count = stop - start
    p_true = records["p_target"][start:stop]
    h_true = records["h_observed"][start:stop]
    p_pred = np.empty(count, dtype=np.float64)
    h_pred = np.empty(count, dtype=np.float64)
    for left in range(start, stop, batch_size):
        right = min(left + batch_size, stop)
        batch = records[left:right]
        score = (weights[0] * batch["x_correct"]
                 + weights[1] * batch["x_wrong"] + weights[2])
        if len(weights) > 3:
            columns = batch["lexeme_column"]
            known = (columns >= 3) & (columns < len(weights))
            score[known] += weights[columns[known]]
        h_batch = np.exp2(np.clip(score, LOG_MIN_H, LOG_MAX_H))
        p_batch = np.clip(np.exp2(-batch["delta_days"] / h_batch), MIN_P, MAX_P)
        h_pred[left-start:right-start] = h_batch
        p_pred[left-start:right-start] = p_batch

    p_error = p_pred - p_true
    mae_p = float(np.abs(p_error).mean())
    p_sq = float(np.dot(p_error, p_error))
    del p_error
    h_error = h_pred - h_true
    h_sq = float(np.dot(h_error, h_error))
    del h_error
    loss_sum = p_sq + ALPHA * h_sq + L2 * float(np.dot(weights, weights))
    auc = auc_fixed(np.rint(p_true), p_pred)
    spearman = spearman_with_ties(h_true, h_pred)
    return {
        "split": split_name, "rows": count, "loss_total": loss_sum,
        "loss_per_row": loss_sum / count, "MAE(p)": mae_p,
        "AUC": auc, "Spearman(h)": spearman,
    }

def show_model_result(name, weights):
    print(f"{name}；输入特征维度：{len(weights):,}")
    print("超参数：", {"alpha": ALPHA, "L2": L2, "learning_rate": LEARNING_RATE,
                       "epochs": EPOCHS, "seed": SEED})
    print("Loss = Σ[(p−p̂)² + 0.01(h−ĥ)²] + 0.1||θ||²")
    print("训练已完成；下表分别是完整 Train、Validation 和 Test 的效果：")
    result = pd.DataFrame([evaluate_hlr(weights, name) for name in splits])
    display(result.round({"loss_total": 3, "loss_per_row": 5,
                          "MAE(p)": 6, "AUC": 6, "Spearman(h)": 6}))
    return result

# === ORIGINAL CELL 038 ===
import csv
import gzip

CATEGORY_DTYPE = np.dtype([
    ("user_index", "<i4"),
    ("learning_index", "i1"),
    ("ui_index", "i1"),
])
category_path = cache_dir / "takeaway_hlr_categories_v1.dat"
category_manifest_path = cache_dir / "takeaway_hlr_categories_v1.json"
category_identity = {
    "version": 1,
    "source_sha256": source_sha256,
    "n_rows": n_rows,
    "train_end": train_end,
    "dtype": repr(CATEGORY_DTYPE.descr),
}
category_manifest = (
    json.loads(category_manifest_path.read_text(encoding="utf-8"))
    if category_manifest_path.is_file() else {}
)
category_valid = (
    category_path.is_file()
    and category_path.stat().st_size == n_rows * CATEGORY_DTYPE.itemsize
    and all(category_manifest.get(k) == v for k, v in category_identity.items())
    and category_manifest.get("user_count", 0) > 0
    and len(category_manifest.get("learning_values", [])) > 0
    and len(category_manifest.get("ui_values", [])) > 0
)
if not category_valid:
    category_vocab = {
        "user_id": {},
        "learning_language": {},
        "ui_language": {},
    }
    partial_path = category_path.with_suffix(".partial")
    category_out = np.memmap(
        partial_path, dtype=CATEGORY_DTYPE, mode="w+", shape=(n_rows,)
    )
    count = 0
    with gzip.open(data_path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = [header.index(name) for name in category_vocab]
        for row_index, row in enumerate(reader):
            if row_index >= n_rows:
                raise ValueError("Raw file has more rows than the base cache")
            values = [row[column] for column in columns]
            if row_index < train_end:
                encoded = [
                    category_vocab[name].setdefault(value, len(category_vocab[name]))
                    for name, value in zip(category_vocab, values)
                ]
            else:
                encoded = [
                    category_vocab[name].get(value, -1)
                    for name, value in zip(category_vocab, values)
                ]
            if encoded[1] > 127 or encoded[2] > 127:
                raise ValueError("Language category count exceeds int8 range")
            category_out[row_index] = tuple(encoded)
            count += 1
    if count != n_rows:
        raise ValueError(f"Category rows {count:,} != base rows {n_rows:,}")
    category_out.flush()
    del category_out
    os.replace(partial_path, category_path)
    category_manifest = {
        **category_identity,
        "user_count": len(category_vocab["user_id"]),
        "learning_values": list(category_vocab["learning_language"]),
        "ui_values": list(category_vocab["ui_language"]),
    }
    category_manifest_path.write_text(
        json.dumps(category_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Created Train-only categorical index cache.")
else:
    print("Reused Train-only categorical index cache.")

categorical_records = np.memmap(
    category_path, dtype=CATEGORY_DTYPE, mode="r", shape=(n_rows,)
)
n_users = int(category_manifest["user_count"])
n_learning_languages = len(category_manifest["learning_values"])
n_ui_languages = len(category_manifest["ui_values"])
assert np.all(categorical_records["user_index"][:train_end] >= 0)
assert np.all(categorical_records["learning_index"][:train_end] >= 0)
assert np.all(categorical_records["ui_index"][:train_end] >= 0)
assert n_users == 92_392 and n_learning_languages == 6 and n_ui_languages == 4
print(
    f"Train categories: {n_users:,} users, {n_learning_languages} learning "
    f"languages, {n_ui_languages} UI languages."
)
print(
    f"Validation unseen users: "
    f"{np.count_nonzero(categorical_records['user_index'][train_end:test_start] < 0):,}"
)

