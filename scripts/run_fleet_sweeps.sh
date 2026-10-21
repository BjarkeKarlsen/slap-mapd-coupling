#!/bin/bash
# Run Token Passing load sweeps for all fleet sizes (8, 16, 32, 64 agents)
# Usage: ./scripts/run_fleet_sweeps.sh

set -e

REPO=$(cd "$(dirname "$0")" && cd .. && pwd)
cd "$REPO"

echo "========================================================================"
echo "Token Passing Load Sweeps for All Fleet Sizes"
echo "========================================================================"
echo ""

# Configuration
# Rates scaled by fleet size (direct multiplication for comparability)
# Base: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
declare -A RATES_BY_SIZE
RATES_BY_SIZE[8]="0.2 0.3 0.4 0.5 0.6 0.7 0.8"
RATES_BY_SIZE[16]="0.4 0.6 0.8 1.0 1.2 1.4 1.6"
RATES_BY_SIZE[32]="0.8 1.2 1.6 2.0 2.4 2.8 3.2"
RATES_BY_SIZE[64]="1.6 2.4 3.2 4.0 4.8 5.6 6.4"

SEEDS=6
VENV="${HOME}/.venvs/rl/bin"

# Check if venv exists
if [ ! -f "$VENV/python" ]; then
    echo "Error: Virtual environment not found at $VENV"
    echo "Please ensure ~/.venvs/rl is set up with the project dependencies"
    exit 1
fi

# Create output directory
mkdir -p DIR

# Run sweeps for each fleet size
for SIZE in 8 16 32 64; do
    CONFIG="configs/fleet_$SIZE/centralised.yaml"
    OUTPUT_DIR="DIR/sweep_${SIZE}_agents"
    RATES="${RATES_BY_SIZE[$SIZE]}"

    if [ ! -f "$CONFIG" ]; then
        echo "✗ Config not found: $CONFIG"
        continue
    fi

    echo "========== Fleet size: $SIZE agents =========="
    echo "Config: $CONFIG"
    echo "Output: $OUTPUT_DIR"
    echo "Rates: $RATES"
    echo ""

    $VENV/python scripts/load_sweep.py \
        --config "$CONFIG" \
        --rates $RATES \
        --seeds $SEEDS \
        --storage-modes fixed demand congestion \
        --out "$OUTPUT_DIR"

    echo "✓ Completed sweep for $SIZE agents"
    echo ""
done

echo "========================================================================"
echo "All sweeps completed!"
echo "========================================================================"
echo ""
echo "Results in:"
echo "  DIR/sweep_8_agents/"
echo "  DIR/sweep_16_agents/"
echo "  DIR/sweep_32_agents/"
echo "  DIR/sweep_64_agents/"
echo ""
echo "Each contains:"
echo "  - load_sweep.jsonl (raw results)"
echo "  - summary.md (human-readable summary)"
echo ""
echo "Next: extract parameters with:"
echo "  python scripts/extract_sweep_parameters.py"
