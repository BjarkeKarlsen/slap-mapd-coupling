#!/usr/bin/env bash
# The sensitivity sweeps at the main load. One Token Passing sweep per demand, arrival and
# storage-rule setting, each changing only that setting over its values (load_sweep.py --vary),
# every other setting as the fleet's config has it. The settings run at once, the fleets
# one after another. RESULTS sets where the results go.
#
#     scripts/sweep/parameter_sweep.bash [FLEET ...]
#
# FLEET is a fleet size of configs/fleet_<FLEET>/decentralised.yaml (default 8 16 32 64). Each
# fleet runs at the arrival rate RATE when it is set, else at the main load its rate sweep found
# (results/arrival_rates/load_fleet<FLEET>/capacity.json). A fleet with neither is skipped. A
# sweep started again skips the rows it already has.
set -uo pipefail
cd "$(dirname "$0")/../.."  # the repo root, where scripts/ and configs/ are

PYTHON=~/.venvs/rl/bin/python
RESULTS=${RESULTS:-~/Documents/thesis/thesis-progress/results}
SEEDS=${SEEDS:-50}

# section.key | storage modes | values. The main result is the gain of demand over fixed storage,
# so a demand or stock setting runs under both. Delta is also the arrival period, so it runs under
# every rule. beta acts only under F_cng, and its F_dem reference is the epoch_length row at the
# config's Delta. The demand settings are fixed by the thesis ([A34]), so their sweep is a
# robustness check only. cold_stock_top_k changes a measure, not a run, so it is not swept.
SWEEPS=(
    "arrival.margin|fixed demand|0.25 0.5 1.0 1.5 2.0"
    "storage_rule.epoch_length|fixed demand congestion|50 100 200 400"
    "storage_rule.congestion_weight|congestion|0.0 0.5 1.0 2.0 3.0 4.0"
    "demand.skew|fixed demand|0.5 1.0 1.5"
    "demand.rotation_period|fixed demand|100 200 400"
    "demand.rotation_shift|fixed demand|2 5 10"
    # "arrival.cold_stock_top_k|fixed|5 10 20 30"
)

main_load() {
    "$PYTHON" -c 'import json, sys
load = json.load(open(sys.argv[1]))["by_mode"]["fixed"]["main_load"]
print("" if load is None else load)' "$RESULTS/arrival_rates/load_fleet$1/capacity.json" 2>/dev/null
}

[ "$#" -eq 0 ] && set -- 8 16 32 64
failed=0
for fleet in "$@"; do
    config=configs/fleet_$fleet/decentralised.yaml
    rate=${RATE:-$(main_load "$fleet")}
    if [ -z "$rate" ]; then
        echo "fleet $fleet skipped, no main load in its capacity.json and no RATE" >&2
        failed=1
        continue
    fi
    out=$RESULTS/parameter_sweep/fleet_$fleet
    echo "fleet $fleet at arrival rate $rate"
    for sweep in "${SWEEPS[@]}"; do
        IFS='|' read -r setting modes values <<< "$sweep"
        dir=$out/$setting
        mkdir -p "$dir"
        cp "$config" "$dir/config.yaml"  # the config the values were varied from
        echo "  start $setting ($modes): $values"
        # shellcheck disable=SC2086  # modes and values are word lists
        "$PYTHON" scripts/sweep/load_sweep.py --config "$config" --rates "$rate" \
            --seeds "$SEEDS" --storage-modes $modes --vary "$setting" --values $values \
            --out "$dir" > "$dir/sweep.log" 2>&1 &
    done
    for job in $(jobs -p); do
        wait "$job" || failed=1
    done
    echo "fleet $fleet done"
done
if [ "$failed" -ne 0 ]; then
    echo "Something failed, see $RESULTS/parameter_sweep/fleet_*/*/sweep.log" >&2
    exit 1
fi
echo "All sweeps done in $RESULTS/parameter_sweep"
