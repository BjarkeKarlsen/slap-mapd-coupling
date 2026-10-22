#!/usr/bin/env bash
# Merge the four fleet folders of z.bash into one summary.md, instance_report.jsonl and
# instances_used.json, the files the graph appendix and warehouses.py read.
set -euo pipefail
cd "$(dirname "$0")/../.."  # the repo root, where scripts/ is

~/.venvs/rl/bin/python -c '
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from instance_report import summary
out = Path.home() / "thesis_runs/instance_report_2026-10-21"
rows, used = [], {}
for m in (8, 16, 32, 64):
    d = out / f"fleet_{m}"
    rows += [json.loads(line) for line in (d / "instance_report.jsonl").open()]
    used.update(json.loads((d / "instances_used.json").read_text()))
(out / "instance_report.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
(out / "instances_used.json").write_text(json.dumps(used))
(out / "summary.md").write_text(summary(rows, {int(m): u["seed"] for m, u in used.items()}))
print(f"Merged {len(rows)} instances into {out}")
'
