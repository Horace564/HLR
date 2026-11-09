"""Check whether history_seen decreases for a user/lexeme at a later timestamp.

Rows are grouped by (user_id, learning_language, lexeme_id), then ordered by
timestamp. Equal timestamps have no known order and are never compared with
one another. Disk buckets keep the full 12.9M-row input out of memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/settles.acl16.learning_traces.13m.csv.gz"
DEFAULT_REPORT = ROOT / "reports/history_seen_consistency.md"
DEFAULT_JSON = ROOT / "reports/history_seen_consistency.json"
DEFAULT_VIOLATIONS = ROOT / "data/interim/history_seen_decreases.csv.gz"
DEFAULT_AUDIT = ROOT / "reports/data_quality.json"
REQUIRED_COLUMNS = {
    "user_id", "learning_language", "lexeme_id", "timestamp", "history_seen"
}
VIOLATION_COLUMNS = (
    "user_id", "learning_language", "lexeme_id",
    "earlier_row_index", "earlier_timestamp", "earlier_history_seen",
    "later_row_index", "later_timestamp", "later_history_seen", "drop",
)
BUCKET_COUNT = 256


def bucket_number(user_id: str, language: str, lexeme_id: str) -> int:
    key = (user_id, language, lexeme_id)
    return hashlib.blake2b(repr(key).encode("utf-8"), digest_size=1).digest()[0]


def inspect_ordered_rows(
    rows: list[tuple[str, str, str, int, int, int]],
    writer: csv.writer,
    examples: list[dict],
    *,
    example_limit: int = 20,
) -> dict[str, int]:
    """Inspect one bucket after sorting by key, timestamp, and raw row index."""
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[5]))
    counts = {
        "distinct_user_lexeme_pairs": 0,
        "distinct_pair_timestamps": 0,
        "pairs_with_decrease": 0,
        "later_rows_with_decrease": 0,
        "timestamp_groups_with_decrease": 0,
        "same_timestamp_groups_with_different_history_seen": 0,
        "largest_drop": 0,
    }
    for key, pair_iter in itertools.groupby(rows, key=lambda row: row[:3]):
        counts["distinct_user_lexeme_pairs"] += 1
        prior_max = None
        pair_has_decrease = False
        for timestamp, time_iter in itertools.groupby(pair_iter, key=lambda row: row[3]):
            group = list(time_iter)
            counts["distinct_pair_timestamps"] += 1
            seen_values = {row[4] for row in group}
            if len(seen_values) > 1:
                counts["same_timestamp_groups_with_different_history_seen"] += 1

            group_has_decrease = False
            for later in group:
                if prior_max is None or later[4] >= prior_max[4]:
                    continue
                drop = prior_max[4] - later[4]
                counts["later_rows_with_decrease"] += 1
                counts["largest_drop"] = max(counts["largest_drop"], drop)
                pair_has_decrease = True
                group_has_decrease = True
                record = {
                    "user_id": key[0],
                    "learning_language": key[1],
                    "lexeme_id": key[2],
                    "earlier_row_index": prior_max[5],
                    "earlier_timestamp": prior_max[3],
                    "earlier_history_seen": prior_max[4],
                    "later_row_index": later[5],
                    "later_timestamp": timestamp,
                    "later_history_seen": later[4],
                    "drop": drop,
                }
                writer.writerow(record[column] for column in VIOLATION_COLUMNS)
                if len(examples) < example_limit:
                    examples.append(record)
            if group_has_decrease:
                counts["timestamp_groups_with_decrease"] += 1

            group_max = max(group, key=lambda row: row[4])
            if prior_max is None or group_max[4] > prior_max[4]:
                prior_max = group_max
        if pair_has_decrease:
            counts["pairs_with_decrease"] += 1
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_history_seen(
    input_path: Path,
    violations_path: Path,
    *,
    max_rows: int | None = None,
) -> dict:
    input_path = input_path.resolve()
    violations_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(tempfile.gettempdir()).resolve()
    total_rows = 0
    examples: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="hlr_history_", dir=scratch_root) as temp_name:
        temp_path = Path(temp_name).resolve()
        if scratch_root not in temp_path.parents:
            raise RuntimeError(f"Temporary directory escaped its root: {temp_path}")
        bucket_paths = [temp_path / f"{number:02x}.csv" for number in range(BUCKET_COUNT)]
        with ExitStack() as stack:
            writers = [
                csv.writer(stack.enter_context(bucket.open("w", encoding="utf-8", newline="")))
                for bucket in bucket_paths
            ]
            with gzip.open(input_path, "rt", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
                    raise ValueError(f"Required columns missing: {sorted(REQUIRED_COLUMNS)}")
                for row_index, row in enumerate(reader):
                    if max_rows is not None and row_index >= max_rows:
                        break
                    user_id = row["user_id"]
                    language = row["learning_language"]
                    lexeme_id = row["lexeme_id"]
                    writers[bucket_number(user_id, language, lexeme_id)].writerow(
                        (user_id, language, lexeme_id, row["timestamp"],
                         row["history_seen"], row_index)
                    )
                    total_rows += 1

        totals = {
            "distinct_user_lexeme_pairs": 0,
            "distinct_pair_timestamps": 0,
            "pairs_with_decrease": 0,
            "later_rows_with_decrease": 0,
            "timestamp_groups_with_decrease": 0,
            "same_timestamp_groups_with_different_history_seen": 0,
            "largest_drop": 0,
        }
        with gzip.open(violations_path, "wt", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(VIOLATION_COLUMNS)
            for bucket_path in bucket_paths:
                with bucket_path.open("r", encoding="utf-8", newline="") as source:
                    records = [
                        (user, language, lexeme, int(ts), int(seen), int(index))
                        for user, language, lexeme, ts, seen, index in csv.reader(source)
                    ]
                bucket_counts = inspect_ordered_rows(records, writer, examples)
                for name, count in bucket_counts.items():
                    if name == "largest_drop":
                        totals[name] = max(totals[name], count)
                    else:
                        totals[name] += count

    result = {
        "input_file": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": sha256_file(input_path),
        "scope": "full_file" if max_rows is None else f"first_{max_rows}_rows",
        "rows_scanned": total_rows,
        "group_key": ["user_id", "learning_language", "lexeme_id"],
        "comparison": "later timestamp > earlier timestamp and later history_seen < maximum earlier history_seen",
        "equal_timestamp_policy": "not ordered; differences counted separately",
        "violations_file": str(violations_path.resolve()),
        **totals,
        "examples": examples,
    }
    return result


def render_report(result: dict) -> str:
    lines = [
        "# 用户 × 词项历史次数的时间一致性检查",
        "",
        f"- 范围：`{result['scope']}`；原始文件 SHA-256：`{result['input_sha256']}`",
        f"- 读取行数：**{result['rows_scanned']:,}**；用户 × 学习语言 × `lexeme_id` 组合：**{result['distinct_user_lexeme_pairs']:,}**。",
        f"- 同一组合的不同时间戳组：**{result['distinct_pair_timestamps']:,}**。",
        "",
        "## 判定方法",
        "",
        "对每个 `(user_id, learning_language, lexeme_id)`，按 `timestamp` 升序排列。若较晚一行的 `history_seen` 小于任何更早时间戳行的值，则该较晚行记为违规。实际实现用之前时间组的最大 `history_seen` 比较，因而能发现非相邻时间点的下降。原始文件的行顺序不参与判定。",
        "",
        "同一秒的行没有可确定的先后顺序：只统计其 `history_seen` 是否不同，不在同秒内部判定下降。完整重复行按此前决定保留；它们不会凭相同时间戳和相同计数形成违规。",
        "",
        "## 结果",
        "",
        "| 指标 | 数量 |",
        "|---|---:|",
        f"| 出现下降的用户 × 词项组合 | {result['pairs_with_decrease']:,} |",
        f"| 出现下降的较晚记录 | {result['later_rows_with_decrease']:,} |",
        f"| 出现下降的较晚时间戳组 | {result['timestamp_groups_with_decrease']:,} |",
        f"| 最大下降次数 | {result['largest_drop']:,} |",
        f"| 同秒但 `history_seen` 不同的组（另列，不算下降） | {result['same_timestamp_groups_with_different_history_seen']:,} |",
        "",
        f"全部下降记录及对应较早记录保存在 `{result['violations_file']}`。行号是原始 CSV 数据区的零基行号，不含表头。以下最多展示前 20 个样例（按分桶顺序，并非原始文件顺序）：",
        "",
        "| 用户 | 语言 | 词项 ID | 较早行号 / 时间戳 / seen | 较晚行号 / 时间戳 / seen | 下降 |",
        "|---|---|---|---|---|---:|",
    ]
    if result["examples"]:
        for ex in result["examples"]:
            lines.append(
                f"| `{ex['user_id']}` | `{ex['learning_language']}` | "
                f"`{ex['lexeme_id']}` | {ex['earlier_row_index']:,} / "
                f"{ex['earlier_timestamp']} / {ex['earlier_history_seen']} | "
                f"{ex['later_row_index']:,} / {ex['later_timestamp']} / "
                f"{ex['later_history_seen']} | {ex['drop']:,} |"
            )
    else:
        lines.append("| 无 | — | — | — | — | 0 |")
    lines.extend([
        "",
        "本脚本只审计并记录，不删除或改写任何原始记录。重新运行：`python src/check_history_seen.py`。它仅依赖 Python 标准库，按哈希分桶暂存必要列，处理完成后清除临时文件。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--violations", type=Path, default=DEFAULT_VIOLATIONS)
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke test only.")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")

    result = check_history_seen(args.input, args.violations, max_rows=args.max_rows)
    if DEFAULT_AUDIT.is_file() and args.input.resolve() == DEFAULT_INPUT.resolve():
        audit = json.loads(DEFAULT_AUDIT.read_text(encoding="utf-8"))
        if result["input_sha256"] != audit["input_sha256"]:
            raise ValueError("Raw file SHA-256 differs from CP2 audit.")
        if args.max_rows is None and result["rows_scanned"] != audit["total_rows"]:
            raise ValueError("Row count differs from CP2 audit.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Scanned {result['rows_scanned']:,} rows; "
        f"{result['later_rows_with_decrease']:,} later rows decreased "
        f"across {result['pairs_with_decrease']:,} user/lexeme pairs."
    )
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
