#!/bin/bash
# Run the Token Passing load sweep for every fleet size (8, 16, 32, 64 agents), fixed storage only.
# Usage: ./scripts/run_fleet_sweeps.sh [OUT]
#
# The rates are given per agent and multiplied by the fleet size m in load_sweep.py, so one grid
# serves every fleet. Each sweep writes OUT/sweep_<m>_agents/capacity.json, the highest rate that
# keeps up on every seed and the main load at 0.6 of it. Add rates between the capacity and the
# first failing rate with --rates-per-agent to the same OUT, and only the new rows run.

set -e

REPO=$(cd "$(dirname "$0")" && cd .. && pwd)
cd "$REPO"

OUT="${1:-DIR}"
RATES_PER_AGENT="0.0125 0.025 0.0375 0.05 0.0625 0.075 0.0875 0.1"
SEEDS=6
VENV="${HOME}/.venvs/rl/bin"

if [ ! -f "$VENV/python" ]; then
    echo "Error: virtual environment not found at $VENV"
    exit 1
fi

mkdir -p "$OUT"

for SIZE in 8 16 32 64; do
    CONFIG="configs/fleet_$SIZE/centralised.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "Config not found: $CONFIG"
        continue
    fi
    echo "========== Fleet size: $SIZE agents =========="
    $VENV/python scripts/load_sweep.py \
        --config "$CONFIG" \
        --rates-per-agent $RATES_PER_AGENT \
        --seeds $SEEDS \
        --out "$OUT/sweep_${SIZE}_agents"
done

echo "Next: python scripts/extract_sweep_parameters.py $OUT"
