"""Run the four put-away rules of the arrival stock model on the 8-agent warehouse and one seed.

Under the arrival stock model stock arrives every Delta steps and the storage rule only decides
which shelves it goes to (storage/arrival.py). The amounts are the same under every rule, so the
rules differ in where the stock sits, which this script prints: how many shelves hold stock and
the mean access distance of the units, from the shelf to the nearest delivery vertex. The
warehouse and its settings are those of configs/fleet_8, with Token Passing routing.

"random" draws each shelf uniformly among the shelves with room. It is a check on "demand", not
a rule of the comparison: if "demand" beats it, the gain comes from where demand puts stock and
not only from spreading it.

Run from the repo root: python examples/02b_putaway_rules.py
"""

from pathlib import Path

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.storage.demand import delivery_vertices
from slap_mapd_coupling.training.spec import build_experiment_config, build_study, load_spec

ROOT = Path(__file__).resolve().parents[1]
SEED = 200_000
RULES = ("fixed", "demand", "congestion", "random")


def main() -> None:
    spec = load_spec(ROOT / "configs" / "fleet_8" / "decentralised.yaml")
    study = build_study(spec)
    graph = study.instance.graph
    delivery = delivery_vertices(graph)

    for mode in RULES:
        config = build_experiment_config(spec, SEED, mode)
        config = ExperimentConfig.model_validate(
            {**config.model_dump(), "controller": None, "planner": "token_passing"}
        )
        env, step = _planner_run(
            graph, study.skus, study.counts, study.capacities, config, "token_passing"
        )
        for _ in range(3 * config.storage_epoch_length):
            step()
        units = 0
        distance = 0.0
        for per_vertex in env.storage.counts.values():
            for v, count in per_vertex.items():
                if count > 0:
                    units += count
                    distance += count * min(graph.distance(v, g) for g in delivery)
        shelves = sum(1 for v in env.storage.capacities if env.storage.used_capacity(v) > 0)
        print(
            f"{mode:>10}: {units} units on {shelves} shelves, "
            f"mean access distance {distance / units:.2f}, "
            f"arrived {env.log.arrived_units_by_update}"
        )


if __name__ == "__main__":
    main()
