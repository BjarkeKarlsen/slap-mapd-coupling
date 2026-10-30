#!/usr/bin/env bash
# The Token Passing sweeps of every fleet, unattended: the load sweep that sets each fleet's main
# load, then the sensitivity sweeps at that load. Not in the thesis.
#
#     nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \
#         scripts/sweep/overnight.bash > ~/Documents/thesis/thesis-progress/results/overnight.log 2>&1 &
#
# 1. The load sweep of fleets 8, 16, 32 and 64 at once, fixed storage only, on the grid of [A58]
#    (0.0125 m to 0.1 m in eight steps), LOAD_SEEDS seeds per rate (default 20). RATES_<m>, for
#    instance RATES_64="0.5 0.6", adds rates for fleet m where an earlier sweep put its capacity.
# 2. Twice, three more rates between the last passing and the first failing rate of each fleet,
#    or below the lowest rate when that already fails, so the capacity is read on a finer grid
#    ([A58]). A fleet whose two rates are a quarter of a grid step apart is not refined again.
#    The load sweep stops at a rate no seed keeps up at, and at one whose starting stock no
#    longer fits on the shelves.
# 3. The main load of each fleet, 0.6 of its capacity, from its capacity.json. A fleet without
#    one is skipped, never given a rate by hand.
# 4. The sensitivity sweeps (parameter_sweep.bash) of fleets 8, 16 and 32 at once, each at its
#    own main load, SEEDS seeds per value (default 20), then fleet 64 on its own.
# The keep-up check is the one of [A59], picks and put-aways separately. Every step skips the rows
# it already has, so the script can be started again. RESULTS sets where the results go, FLEETS
# which fleets run (default "8 16 32 64"). CONFIGS is the config of a fleet with {} for its size
# (default configs/fleet_{}/decentralised.yaml), SWEEP_ARGS more load_sweep.py options for every
# sweep (for example "--horizon 10000 --warmup 5000"), and WORKERS the processes each load sweep
# of step 1 and 2 plays its seeds in (default 1).
set -uo pipefail
cd "$(dirname "$0")/../.."  # the repo root, where scripts/ and configs/ are

PYTHON=~/.venvs/rl/bin/python
RESULTS=${RESULTS:-~/Documents/thesis/thesis-progress/results}
LOAD_SEEDS=${LOAD_SEEDS:-20}
CONFIGS=${CONFIGS:-configs/fleet_\{\}/decentralised.yaml}
SWEEP_ARGS=${SWEEP_ARGS:-}
WORKERS=${WORKERS:-1}
export RESULTS SEEDS=${SEEDS:-20} CONFIGS SWEEP_ARGS
read -r -a FLEETS <<< "${FLEETS:-8 16 32 64}"
GRID="0.0125 0.025 0.0375 0.05 0.0625 0.075 0.0875 0.1"

log() { echo "$(date '+%d/%m %H:%M') $*"; }
folder() { echo "$RESULTS/arrival_rates/load_fleet$1"; }
config() { echo "${CONFIGS//\{\}/$1}"; }

# capacity FLEET KEY prints one value of the fixed rule's capacity.json, empty when unset.
capacity() {
    "$PYTHON" -c 'import json, sys
found = json.load(open(sys.argv[1]))["by_mode"]["fixed"].get(sys.argv[2])
print("" if found is None else found)' "$(folder "$1")/capacity.json" "$2" 2>/dev/null
}

load_sweep() {  # load_sweep FLEET ARGS...
    local fleet=$1
    shift
    mkdir -p "$(folder "$fleet")"
    # shellcheck disable=SC2086  # SWEEP_ARGS is a word list
    "$PYTHON" scripts/sweep/load_sweep.py --config "$(config "$fleet")" --seeds "$LOAD_SEEDS" \
        --stop-when-overloaded --workers "$WORKERS" $SWEEP_ARGS --out "$(folder "$fleet")" "$@" \
        >> "$(folder "$fleet")/sweep.log" 2>&1
}

log "step 1, load sweeps of fleets ${FLEETS[*]}, $LOAD_SEEDS seeds per rate"
for fleet in "${FLEETS[@]}"; do
    (
        # shellcheck disable=SC2086  # GRID is a word list
        load_sweep "$fleet" --rates-per-agent $GRID
        # RATES_<m>, more rates where an earlier sweep put the capacity, run after the grid.
        extra_name="RATES_$fleet"
        extra=${!extra_name:-}
        if [ -n "$extra" ]; then
            log "fleet $fleet, extra rates $extra"
            # shellcheck disable=SC2086  # extra is a word list
            load_sweep "$fleet" --rates $extra
        fi
    ) &
done
wait

for pass in 1 2; do
    log "step 2, finer rates around each capacity, pass $pass"
    for fleet in "${FLEETS[@]}"; do
        rates=$("$PYTHON" -c 'import sys
low, high, m = sys.argv[1], sys.argv[2], int(sys.argv[3])
if high == "":
    print("")  # every rate kept up, or the sweep has no rows
elif low == "":
    high = float(high)  # the lowest rate already fails
    print(" ".join(f"{round(high * k / 4, 6):g}" for k in (1, 2, 3)))
elif float(high) - float(low) <= 0.0125 * m / 4 + 1e-9:
    print("")  # already a quarter of a grid step apart
else:
    low, high = float(low), float(high)
    print(" ".join(f"{round(low + (high - low) * k / 4, 6):g}" for k in (1, 2, 3)))
' "$(capacity "$fleet" capacity)" "$(capacity "$fleet" first_failing_rate)" "$fleet")
        if [ -n "$rates" ]; then
            log "fleet $fleet, rates $rates"
            # shellcheck disable=SC2086  # rates is a word list
            load_sweep "$fleet" --rates $rates &
        fi
    done
    wait
done

log "step 3, main loads"
declare -A LOAD
for fleet in "${FLEETS[@]}"; do
    LOAD[$fleet]=$(capacity "$fleet" main_load)
    log "fleet $fleet capacity $(capacity "$fleet" capacity), main load ${LOAD[$fleet]:-none}"
done

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

log "step 4, sensitivity sweeps of every fleet but 64 at once, $SEEDS seeds per value"
for fleet in "${FLEETS[@]}"; do
    [ "$fleet" = 64 ] || parameter_sweep "$fleet"
done
wait
if [[ " ${FLEETS[*]} " == *" 64 "* ]]; then
    log "step 4, sensitivity sweep of fleet 64 on its own, $SEEDS seeds per value"
    parameter_sweep 64
    wait
fi
log "done, see $RESULTS/parameter_sweep_fleet*.log"
