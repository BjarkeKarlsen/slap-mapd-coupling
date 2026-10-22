#!/usr/bin/env bash
# Generate the warehouse of every fleet size over 50 seeds, the four fleets in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."  # the repo root, where scripts/ and configs/ are

PYTHON=~/.venvs/rl/bin/python
OUT=~/thesis_runs/instance_report_2026-10-21

for n in 8 16 32 64; do
    mkdir -p "$OUT/fleet_$n"
    "$PYTHON" scripts/instance_report.py \
        --configs "configs/fleet_$n/centralised.yaml" \
        --seeds 50 --out "$OUT/fleet_$n" > "$OUT/fleet_$n.log" 2>&1 &
done

failed=0
for job in $(jobs -p); do
    wait "$job" || failed=1
done
if [ "$failed" -ne 0 ]; then
    echo "A fleet failed, see $OUT/fleet_*.log" >&2
    exit 1
fi
echo "All four fleets done in $OUT"
