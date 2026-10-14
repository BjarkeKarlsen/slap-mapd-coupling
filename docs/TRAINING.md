# Training and evaluating a learned arm

The three learned arms (centralised, section-based, decentralised) train through one
RLlib adapter, `training/env.py::WarehouseMAPDMultiAgentEnv`, with one agent per robot
and one shared policy. `examples/04_decentralised_training.py` is the worked example,
and `02_centralised_fixed.py` and `03_section_based.py` are its variants.

## Configuring a run

`training/config.py::build_ppo_config` takes the instance, the experiment config, the
encoder and head configs, `PPOHyperparameters` and the seed split, and returns a
`PPOConfig`. It rejects, before Ray starts, any setting the adapter would reject.

- The credit signal (`PPOHyperparameters.credit_signal`) has no default. The examples
  pass `"per_agent"`, an illustrative choice.
- The training budget, `PPOHyperparameters.training_budget_env_steps`, has no default.
  It is counted in environment steps and is equal across arms. The examples pass an
  illustrative value.
- `GNNEncoderConfig.pooling` has no default. Use `"mean"` for the centralised and
  section arms and `"none"` for the decentralised arm.
- `LocalSubgraphEncodingConfig.max_local_nodes` must be at least the largest window the
  arm uses, because a window is never truncated. That is every movable vertex for the
  centralised arm, the largest field of view for the decentralised arm, and the largest
  zone with its halo for the section arm. The examples compute it from the graph.
- The section arm needs `zones`, for example `environment/zones.py::zones_by_aisle`
  with `GeneratedInstance.aisle_index`.

## Running it

The examples run the config with Ray Tune, which stops at the first iteration whose
sampled environment steps reach the budget, so a run overshoots by up to one
iteration's samples. `build_ppo_config` sets the rollout length and the train batch from the
horizon, so an iteration samples one episode per runner, `num_env_runners * horizon` steps
(`horizon` steps with no env runner).

```python
from ray import tune

results = tune.Tuner(
    "PPO",
    param_space=ppo_config,  # the PPOConfig object, not to_dict()
    run_config=tune.RunConfig(
        name="my_run",
        stop={"num_env_steps_sampled_lifetime": budget},
        checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=True),
    ),
).fit()
checkpoint = results[0].checkpoint.path
```

Ray writes the results and the checkpoint under `~/ray_results/<name>/<trial>/`.
There is no checkpoint folder of our own. Tune also writes CSV, JSON and TensorBoard
files for every trial.

## Reading the curves

Install the dev extras, which include `tensorboard`, and point it at the trial folder
or at `~/ray_results`.

```bash
tensorboard --logdir ~/ray_results
```

The scalars are tagged `ray/tune/...`. The Ray dashboard on port 8265 shows actors,
tasks and logs, and no training curves.

## Plotting a run

The examples write one `<arm>_training_progress.jsonl` per arm and, when they trace episodes,
one `<arm>_episode_<n>.jsonl` per trace. `viz/training_run.py` turns a run folder into figures
and prints a few numbers for every trace.

```bash
python -m slap_mapd_coupling.viz.training_run --run-dir examples/output --out figures
```

It writes `training_progress.png` (return and its five terms, override rate with standing still,
tasks completed against released), one `trace_<arm>_episode_<n>.png` per trace (tasks over time
and what each agent did at every step) and, from two traces on, `trace_comparison.png`. The
printed numbers are the tasks completed, the longest stretch without a completion, each agent's
share of steps moved, chose to wait or overridden, the most overridden moves after the last
completion, and the steps where two agents tried to swap vertices. A log from an older run
misses some keys, and its panels are left out.

## Evaluating a checkpoint

`examples/04b_evaluate_checkpoint.py --checkpoint=<path>` loads the module with
`RLModule.from_checkpoint` on `<checkpoint>/learner_group/learner/rl_module/shared_policy`.
The path is built from Ray's constants `COMPONENT_LEARNER_GROUP`, `COMPONENT_LEARNER` and
`COMPONENT_RL_MODULE`, and `tests/unit/test_checkpoint_layout.py` fails if a Ray upgrade
moves the layout. This needs no Ray cluster and no registered environment.

The module then runs through `evaluation/evaluator.py::run_episode(policy=ModulePolicy(module), ...)`,
which builds the same adapter that trained it, so the policy sees the observations it saw in
training. The encoding config and pooling must be those the module was trained with. Scores
come from this single-process, deterministic run and not from Tune.

## Regimes

`fixed_only` trains under F_fix. `matched` trains under the storage rule used at evaluation.
The two differ only once the online storage estimates of the demand and congestion rules
exist, so the example shows the switch and the two regimes train alike until then.
