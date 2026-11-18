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
