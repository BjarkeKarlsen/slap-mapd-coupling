## Routing integration

The simulator core (`environment/multi_agent_env.py`, `WarehouseMAPDEnv`) has one
step mode. `step(slots)` takes one action slot per robot, the slot of the action
mask of `environment/spaces.py`, and runs the six stages. Whatever produces the
slots times itself and records the time on the core's log. There is no controller
registry and no `Controller` protocol. Three pieces plug in around the core.

### Task assignment, the `Assigner`

Task assignment (`controllers/assignment.py`, the assignment rule) is shared by
every router. A planner that picks its own tasks, such as Token Passing, replaces
it by implementing the `Assigner` protocol of the same file, `reset(episode_seed)`
and `assign(graph, fleet, tasks, t)`, and is passed to the core as `assigner=`.
The core resets it with the episode seed and asks it for the assignment of every
timestep.

### A non-learned planner, the planner driver

To add a reference planner:

1. Create `src/slap_mapd_coupling/planners/<name>.py`, implementing the `Planner`
   protocol of `planners/base.py`, `plan(graph, fleet, tasks, t)`. Also implement
   the `Assigner` methods if it picks its own tasks.
2. Add it to `planners/registry.py` and to `PlannerName`.
3. Add a unit test under `tests/unit/`.
4. Add an `examples/` script that runs it.

`planners/driver.py` runs any planner from outside the core. `step_with_planner`
plans, converts the actions to slots, records the planning time and steps the core.
`evaluation/evaluator.py::run_episode` does this for a config with `planner` set,
and a planner run never imports Ray.

```python
from slap_mapd_coupling.planners import get_planner
from slap_mapd_coupling.planners.driver import step_with_planner

planner = get_planner("token_passing")
env = WarehouseMAPDEnv(graph, skus, counts, capacities, config, assigner=planner)
step_with_planner(env, planner)
```

### A learned arm, the policy function

The three learned arms (centralised, section-based, decentralised) run through one
RLlib adapter, `training/env.py::WarehouseMAPDMultiAgentEnv`, with one agent per
robot. The arm only decides the window a robot observes and what the readout pools
over. Training and evaluation both build the observations there. The decision
itself is the policy function, `training/policy.py::ModulePolicy`. It takes an
`RLModule`, calls it once for all robots and samples one legal slot per robot from
a seeded stream. It does not know how the module was loaded.

To evaluate a trained module:

```python
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.training.policy import ModulePolicy

metrics = run_episode(
    graph, skus, counts, capacities, config, trace_path,
    policy=ModulePolicy(module),
    encoding_config=encoding_config,   # the caps the module was trained with
    pooling="none",                    # "mean" for the centralised and section arms
    zones=None,                        # the zones, for the section arm only
)
```

`examples/04_decentralised_training.py` shows the whole route, and `02` and `03` are
its variants for the other two arms. Whichever piece is added, conflict resolution
(`resolution/conflict_resolution.py`) runs identically on the resolved joint action,
so a new router never touches the storage rules or the metrics.
