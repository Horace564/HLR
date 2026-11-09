"""Stream the raw HLR CSV and audit deterministic Track B validity rules.

Track A always reads the original file unchanged. Track B excludes only rows
with missing required values, malformed numeric values, or violated numeric
relationships. Duplicate rows and extreme but valid values are reported, not
silently removed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import tempfile
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/settles.acl16.learning_traces.13m.csv.gz"
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


def open_rows(path: Path) -> Iterator[tuple[int, dict[str, str | None]]]:
    """Yield (zero-based data row index, row) in the original file order."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, restkey="_extra", restval=None)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV header: {reader.fieldnames!r}")
        yield from enumerate(reader)


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


def iter_track_a_rows(path: Path = DEFAULT_INPUT) -> Iterator[tuple[int, dict[str, str | None]]]:
    """Raw input for paper comparison, with no CP2 filtering."""
    yield from open_rows(path)


def iter_track_b_rows(path: Path = DEFAULT_INPUT) -> Iterator[tuple[int, dict[str, str | None]]]:
    """Valid rows in original order; row indices always refer to the raw file."""
    for index, row in open_rows(path):
        if not inspect_row(row)[0]:
            yield index, row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(READ_BUFFER), b""):
            digest.update(block)
    return digest.hexdigest()


def row_fingerprint(row: dict[str, str | None]) -> bytes:
    """Fingerprint all fields, including outcomes, for exact-row candidates."""
    ordered = tuple(row.get(field) for field in FIELDS) + (row.get("_extra"),)
    return hashlib.blake2b(repr(ordered).encode("utf-8"), digest_size=16).digest()


def verify_duplicate_candidates(
    path: Path, candidate_digests: set[bytes], max_rows: int | None = None
) -> tuple[int, list[dict[str, int]]]:
    """Compare full field tuples; fingerprints only locate candidate rows."""
    first_rows: dict[bytes, list[tuple[tuple[object, ...], int]]] = {}
    duplicate_count = 0
    examples: list[dict[str, int]] = []
    for index, row in open_rows(path):
        if max_rows is not None and index >= max_rows:
            break
        digest = row_fingerprint(row)
        if digest not in candidate_digests:
            continue
        full_row = tuple(row.get(field) for field in FIELDS) + (row.get("_extra"),)
        seen = first_rows.setdefault(digest, [])
        match = next((first_index for previous, first_index in seen if previous == full_row), None)
        if match is None:
            seen.append((full_row, index))
        else:
            duplicate_count += 1
            if len(examples) < 10:
                examples.append({"first_row_index": match, "duplicate_row_index": index})
    return duplicate_count, examples


def audit(
    path: Path,
    exclusions_path: Path,
    *,
    max_rows: int | None = None,
) -> dict:
    """Audit a gzip CSV without materializing all rows in memory."""
    path = path.resolve()
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    interim_dir = ROOT / "data/interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    missing = Counter()
    reason_counts = Counter()
    unique_values = {field: set() for field in FIELDS}
    numeric_ranges: dict[str, list[int | float | None]] = {
        field: [None, None] for field in NUMERIC_FIELDS
    }
    language_pairs = Counter()
    special = Counter()
    extreme_delta: list[tuple[int, int]] = []
    total = excluded = 0
    previous_timestamp: int | None = None
    candidate_digests: set[bytes] = set()
    fingerprint_repeats = 0

    with tempfile.TemporaryDirectory(prefix="cp2_duplicates_", dir=interim_dir) as temp_name:
        bucket_paths = [Path(temp_name) / f"{number:02x}.bin" for number in range(256)]
        with ExitStack() as stack:
            buckets = [stack.enter_context(bucket.open("wb")) for bucket in bucket_paths]
            with gzip.open(exclusions_path, "wt", encoding="utf-8", newline="") as excluded_file:
                writer = csv.writer(excluded_file)
                writer.writerow(("source_row_index", "reasons") + FIELDS)
                for index, row in open_rows(path):
                    if max_rows is not None and index >= max_rows:
                        break
                    total += 1
                    reasons, numbers = inspect_row(row)

                    for field in FIELDS:
                        raw = row.get(field)
                        if raw is None or not raw.strip():
                            missing[field] += 1
                        else:
                            unique_values[field].add(raw)
                    for field, number in numbers.items():
                        bounds = numeric_ranges[field]
                        bounds[0] = number if bounds[0] is None else min(bounds[0], number)
                        bounds[1] = number if bounds[1] is None else max(bounds[1], number)

                    pair = (row.get("ui_language"), row.get("learning_language"))
                    language_pairs[pair] += 1
                    p = numbers.get("p_recall")
                    if p == 0:
                        special["p_recall_zero"] += 1
                    if p == 1:
                        special["p_recall_one"] += 1
                    delta = numbers.get("delta")
                    if delta is not None:
                        if delta == 0:
                            special["delta_zero"] += 1
                        if delta > 30 * 86400:
                            special["delta_gt_30_days"] += 1
                        if delta > 365 * 86400:
                            special["delta_gt_365_days"] += 1
                        if len(extreme_delta) < 10:
                            heapq.heappush(extreme_delta, (delta, index))
                        elif delta > extreme_delta[0][0]:
                            heapq.heapreplace(extreme_delta, (delta, index))
                    timestamp = numbers.get("timestamp")
                    if timestamp is not None:
                        if timestamp <= 0:
                            special["timestamp_nonpositive"] += 1
                        if previous_timestamp is not None and timestamp < previous_timestamp:
                            special["timestamp_backwards_adjacent"] += 1
                        previous_timestamp = timestamp

                    if reasons:
                        excluded += 1
                        reason_counts.update(reasons)
                        writer.writerow((index, ";".join(reasons)) + tuple(row.get(field) for field in FIELDS))

                    digest = row_fingerprint(row)
                    buckets[digest[0]].write(digest)

        for bucket_path in bucket_paths:
            seen: set[bytes] = set()
            with bucket_path.open("rb") as source:
                for digest in iter(lambda: source.read(16), b""):
                    if digest in seen:
                        fingerprint_repeats += 1
                        candidate_digests.add(digest)
                    else:
                        seen.add(digest)
        # The temporary bucket files are deleted when leaving this block.

    if candidate_digests:
        duplicate_count, duplicate_examples = verify_duplicate_candidates(
            path, candidate_digests, max_rows=max_rows
        )
    else:
        duplicate_count, duplicate_examples = 0, []

    result = {
        "input_file": str(path),
        "input_size_bytes": path.stat().st_size,
        "input_sha256": sha256_file(path),
        "scope": "full_file" if max_rows is None else f"first_{max_rows}_rows",
        "total_rows": total,
        "track_a_rows": total,
        "track_b_retained_rows": total - excluded,
        "track_b_excluded_rows": excluded,
        "ratio_abs_tolerance": RATIO_ABS_TOL,
        "missing_counts": {field: missing[field] for field in FIELDS},
        "distinct_nonmissing_raw_values": {
            field: len(unique_values[field]) for field in FIELDS
        },
        "numeric_ranges": {field: {"min": bounds[0], "max": bounds[1]} for field, bounds in numeric_ranges.items()},
        "rule_hit_counts": {rule: reason_counts[rule] for rule in VALIDITY_RULES},
        "special_counts": {key: special[key] for key in DIAGNOSTIC_COUNTS},
        "language_pairs": [
            {"ui_language": ui, "learning_language": learning, "rows": count}
            for (ui, learning), count in sorted(language_pairs.items(), key=lambda item: (item[1], str(item[0])))
        ],
        "largest_delta_seconds": [
            {"source_row_index": index, "delta_seconds": delta}
            for delta, index in sorted(extreme_delta, reverse=True)
        ],
        "fingerprint_repeat_count": fingerprint_repeats,
        "exact_duplicate_rows_after_first": duplicate_count,
        "duplicate_examples": duplicate_examples,
        "exclusions_file": str(exclusions_path.resolve()),
        "duplicate_policy": "report_only_keep_all_rows",
    }
    assert result["track_b_retained_rows"] + result["track_b_excluded_rows"] == total
    return result


def utc_timestamp(value: int | float | None) -> str:
    if value is None:
        return "无有效值"
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return f"{value}（超出可转换范围）"


def render_report(result: dict) -> str:
    total = result["total_rows"]
    lines = [
        "# CP2 数据质量与清洗记录",
        "",
        f"- 范围：`{result['scope']}`；原始文件：`{result['input_file']}`",
        f"- SHA-256：`{result['input_sha256']}`",
        f"- 原始 / Track A：**{total:,}** 行；Track B 保留：**{result['track_b_retained_rows']:,}** 行；排除：**{result['track_b_excluded_rows']:,}** 行。保留 + 排除 = 原始。",
        f"- Track B 排除明细：`{result['exclusions_file']}`；每行记录原始零基行号和全部命中的原因。",
        "",
        "## 各列概览",
        "",
        "| 字段 | 缺失数 | 缺失率 | 非缺失原始值去重数 | 解析后数值范围 |",
        "|---|---:|---:|---:|---|",
    ]
    for field in FIELDS:
        missing = result["missing_counts"][field]
        bounds = result["numeric_ranges"].get(field)
        span = f"{bounds['min']} ~ {bounds['max']}" if bounds else "文本"
        lines.append(
            f"| `{field}` | {missing:,} | {missing / total:.4%} | "
            f"{result['distinct_nonmissing_raw_values'][field]:,} | {span} |"
        )
    lines.extend([
        "",
        "“去重数”是该列非缺失 CSV 原始文本的精确不同值数量；缺失定义为字段不存在、空串或全空白。数值范围只统计成功解析的有限数值。",
        "",
        "## Track B 有效性规则",
        "",
        f"比例一致性使用绝对容差 `{result['ratio_abs_tolerance']}`，相对容差为 0。各规则命中数可以重叠；排除数按行取并集。",
        "",
        "| 排除原因 | 命中行数 |",
        "|---|---:|",
    ])
    if result["rule_hit_counts"]:
        lines.extend(f"| `{rule}` | {count:,} |" for rule, count in result["rule_hit_counts"].items())
    else:
        lines.append("| 无 | 0 |")
    lines.extend([
        "",
        "规则：任何必需字段缺失或数值类型无效；`p_recall` 超出 [0,1]；`delta` 或历史计数为负；`history_correct > history_seen`；`session_seen <= 0`；`session_correct < 0` 或 `session_correct > session_seen`；以及目标比例与当前 session 计数不一致。对 `p_recall = 0` 或 `1` **不排除**，也不在 CP2 clipping。多个规则同时命中的行仅排除一次。Track A 不应用任何这些规则。",
        "",
        "## 额外诊断（不自动排除）",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
    ])
    for key, count in result["special_counts"].items():
        lines.append(f"| `{key}` | {count:,} |")
    timestamp = result["numeric_ranges"]["timestamp"]
    lines.extend([
        "",
        f"- 时间戳范围（UTC）：{utc_timestamp(timestamp['min'])} 至 {utc_timestamp(timestamp['max'])}。相邻行时间戳逆序次数见上表；这不等同于正式时间切分结论。",
        f"- 完整行重复：**{result['exact_duplicate_rows_after_first']:,}** 条（仅统计第一次出现之后的重复）。先按全部 12 列指纹定位候选，再逐字段比较确认；不按 `(user_id, lexeme_id, timestamp)` 三元组直接判重。重复行暂时保留，避免未经 session 粒度核实就改变官方样本。",
    ])
    if result["duplicate_examples"]:
        lines.extend([
            "- 完整重复样例（原始零基行号；两行的 12 个字段均相同）：",
            "",
            "| 首次出现 | 重复出现 |",
            "|---:|---:|",
        ])
        for example in result["duplicate_examples"]:
            lines.append(
                f"| {example['first_row_index']:,} | {example['duplicate_row_index']:,} |"
            )
    lines.append("- 最大 `delta` 样例（原始零基行号、秒）：")
    for item in result["largest_delta_seconds"]:
        lines.append(f"  - {item['source_row_index']}: {item['delta_seconds']:,}")
    lines.extend([
        "",
        "### 语言组合（由少到多）",
        "",
        "| UI 语言 | 学习语言 | 行数 | 是否少于 1,000 行 |",
        "|---|---|---:|---|",
    ])
    for pair in result["language_pairs"]:
        lines.append(
            f"| `{pair['ui_language']}` | `{pair['learning_language']}` | "
            f"{pair['rows']:,} | {'是' if pair['rows'] < 1000 else '否'} |"
        )
    lines.extend([
        "",
        "## 清洗边界",
        "",
        "- `src/data.py::iter_track_a_rows()` 按原始顺序返回全部行；`iter_track_b_rows()` 只应用以上确定性有效性规则并保留原始零基行号。没有重新排序、去重或修改字段值。",
        "- 计数和排除原因由同一个 `inspect_row()` 函数产生。异常但有效的长间隔、少见语言组合、`delta = 0`、`p_recall = 0/1` 均保留。后续若改变清洗政策，需重新运行并另记版本。",
        "- 本报告是数据检查，不使用任何 `session_*` 字段作预测输入；它们只用于目标一致性检查。",
        "- 在项目根目录运行 `python src/data.py` 可从原始 gzip 文件重新生成本报告、JSON 统计及排除明细；仅需 Python 标准库。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw HLR data and record Track B exclusions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/data_quality.md")
    parser.add_argument("--json", type=Path, default=ROOT / "reports/data_quality.json")
    parser.add_argument("--exclusions", type=Path, default=ROOT / "data/interim/track_b_exclusions.csv.gz")
    parser.add_argument("--max-rows", type=int, default=None, help="Development smoke test only.")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    result = audit(args.input, args.exclusions, max_rows=args.max_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {result['total_rows']:,} rows; Track B excludes {result['track_b_excluded_rows']:,}.")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
