#!/usr/bin/env bash
# Everything after the rate sweeps, unattended, for fleets 8, 16, 32 and 64. Fleet 64 runs last,
# once the other three are done.
#
#     nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \
#         scripts/sweep/overnight.bash > ~/Documents/thesis/thesis-progress/results/overnight.log 2>&1 &
#
# 1. Waits until no load sweep of fleets 8, 16, 32 or 64 is running.
# 2. Reads the main load of each fleet from the capacity.json of its rate sweep (FOLDER_<m>), or,
#    when that has none, the arrival_rate of configs/fleet_<m>/decentralised.yaml.
# 3. Runs fleets 8 and 16 at their main load, into FOLDER_<m>/rate<load>/, 50 seeds.
# 4. Runs the sensitivity sweeps (parameter_sweep.bash) for the three fleets at once, each at
#    its own main load. SEEDS sets the seeds per value (default 20).
# 5. Runs the sensitivity sweep of fleet 64 at its main load, after step 4 has finished.
# Nothing is merged or overwritten, and every step skips the rows it already has, so the script
# can be started again.
set -uo pipefail
cd "$(dirname "$0")/../.."  # the repo root, where scripts/ and configs/ are

PYTHON=~/.venvs/rl/bin/python
RESULTS=${RESULTS:-~/Documents/thesis/thesis-progress/results}
export RESULTS SEEDS=${SEEDS:-20}

# The rate sweep of each fleet, where its capacity.json is.
declare -A FOLDER=(
    [8]=$RESULTS/arrival_rates/load_fleet8
    [16]=$RESULTS/arrival_rates/load_fleet16
    [32]=$RESULTS/arrival_rates/load_fleet32
    [64]=$RESULTS/arrival_rates/load_fleet64
)

log() { echo "$(date '+%d/%m %H:%M') $*"; }

main_load() {
    "$PYTHON" -c 'import json, sys
load = json.load(open(sys.argv[1]))["by_mode"]["fixed"]["main_load"]
print("" if load is None else load)' "${FOLDER[$1]}/capacity.json" 2>/dev/null
}

config_rate() {
    "$PYTHON" -c 'import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["arrival_rate"])' "configs/fleet_$1/decentralised.yaml"
}

log "step 1, waiting for running load sweeps of fleets 8, 16, 32 and 64"
while pgrep -f "python.*load_sweep.py.*fleet_(8|16|32|64)/" > /dev/null; do
    sleep 60
done

log "step 2, main loads"
declare -A LOAD
for fleet in 8 16 32 64; do
    LOAD[$fleet]=$(main_load "$fleet")
    from=${FOLDER[$fleet]}/capacity.json
    if [ -z "${LOAD[$fleet]}" ]; then
        LOAD[$fleet]=$(config_rate "$fleet")
        from=configs/fleet_$fleet/decentralised.yaml
    fi
    log "fleet $fleet main load ${LOAD[$fleet]:-none}, from $from"
done

log "step 3, fleets 8 and 16 at their main load"
for fleet in 8 16; do
    load=${LOAD[$fleet]}
    [ -n "$load" ] || continue
    out=${FOLDER[$fleet]}/rate${load/./_}
    mkdir -p "$out"
    "$PYTHON" scripts/sweep/load_sweep.py --config "configs/fleet_$fleet/decentralised.yaml" \
        --rates "$load" --seeds 50 --out "$out" > "$out/sweep.log" 2>&1 &
done
wait

# The sensitivity sweep of one fleet at its main load, in the background.
parameter_sweep() {
    local load=${LOAD[$1]}
    if [ -z "$load" ]; then
        log "fleet $1 skipped, no main load"
        return
    fi
    RATE=$load scripts/sweep/parameter_sweep.bash "$1" \
        > "$RESULTS/parameter_sweep_fleet$1.log" 2>&1 &
}

log "step 4, sensitivity sweeps of fleets 8, 16 and 32, $SEEDS seeds per value"
for fleet in 8 16 32; do
    parameter_sweep "$fleet"
done
wait

log "step 5, sensitivity sweep of fleet 64, $SEEDS seeds per value"
parameter_sweep 64
wait
log "done, see $RESULTS/parameter_sweep_fleet*.log"
