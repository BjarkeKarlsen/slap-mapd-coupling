# Training and evaluating a learned arm

The three learned arms (centralised, section-based, decentralised) train through one
RLlib adapter, `training/env.py::WarehouseMAPDMultiAgentEnv`, with one agent per robot
and one shared policy. `examples/04_decentralised_training.py` is the worked example,
and `02_centralised_fixed.py` and `03_section_based.py` are its variants.

## Configuring a run

`training/config.py::build_ppo_config` takes the instance, the experiment config, the
encoder and head configs, `PPOHyperparameters` and the seed split, and returns a
`PPOConfig`. It rejects, before Ray starts, any setting the adapter would reject.

- The credit is not a setting. It follows the controller. The centralised arm credits each
  robot with the mean reward of the fleet, the section arm with the mean reward of the robots
  in its zone, and the decentralised arm and the full-view setting with its own reward.
- The training budget, `PPOHyperparameters.training_budget_env_steps`, has no default.
  It is counted in environment steps and is equal across arms. It is set in the YAML file.
- `GNNEncoderConfig.pooling` has no default. Use `"mean"` for the centralised and
  section arms and `"none"` for the decentralised arm.
- `LocalSubgraphEncodingConfig.max_local_nodes` must be at least the largest window the
  arm uses, because a window is never truncated. That is every movable vertex for the
  centralised arm, the largest field of view for the decentralised arm, and the largest
  zone with its halo for the section arm. The examples compute it from the graph.
- The section arm needs `zones`, for example `environment/zones.py::zones_by_aisle`
  with `GeneratedInstance.aisle_index`.

## Running it

`training/trainer.py::train` runs a finished `PPOConfig` with Ray Tune. It stops at the first
iteration whose sampled environment steps reach the budget, so a run overshoots by up to one
iteration's samples. `build_ppo_config` sets the rollout length and the train batch from the
horizon, so an iteration samples one episode per runner, `num_env_runners * horizon` steps
(`horizon` steps with no env runner). The trainer holds no graph. The caller builds the config.

```python
from slap_mapd_coupling.training.trainer import CheckpointSettings, train

run = train(
    ppo_config,  # the PPOConfig object, not to_dict()
    budget_env_steps,
    "my_run",
    settings=CheckpointSettings(frequency=10, num_to_keep=3),
)
run.checkpoint, run.best_checkpoint
```

Ray writes the results and the checkpoints under `~/ray_results/<name>/<trial>/`, or under
`storage_path` when one is given. Tune also writes CSV, JSON and TensorBoard files for every trial.

### Checkpoints and the best one

Checkpointing is Tune's own (`tune.CheckpointConfig`), set by `CheckpointSettings`.

- `frequency` is the number of iterations between checkpoints, and 0 keeps only the one written at
  the end of the run, which is always written.
- `num_to_keep` keeps that many checkpoints, the best by `score_attribute`, and always the latest.
  None keeps all of them.
- `score_attribute` is `episode_return_mean` and `score_order` is `max`, so the best
  checkpoint is the one with the highest training return. The score must be a top-level key of the
  result, so `TrainingProgressCallbacks` copies `env_runners/episode_return_mean` there. Tune ranks
  by a nested key only when it reads the best one back and deletes by recency, which keeps the newest
  checkpoints and not the best, so `CheckpointSettings` rejects a name with a `/`.

`TrainingRun.checkpoint` is the checkpoint at the end of the run and `TrainingRun.best_checkpoint`
the best one that was kept. They are the same folder when the last iteration scored best. The
score is the training return, not an evaluation. A run can raise its return while completing no
tasks (see `deliver_reward` in `configs/decentralised.yaml`), so evaluate the best checkpoint
with `04b_evaluate_checkpoint.py` before trusting it.

### The config file and `slap-mapd train`

One YAML file describes a run: the instance, storage, experiment settings, model sizes, the PPO
hyperparameters, the budget and the checkpointing. `configs/decentralised.yaml`, `centralised.yaml` and `section.yaml` are the files of examples 04, 02
and 03. They differ only in `controller`, and their values are illustrative because the thesis
leaves them TBD. Edit all three to change a value for every arm.
`training/spec.py` loads it and builds everything `train` needs.

```bash
slap-mapd train --config configs/decentralised.yaml --budget-env-steps 4200 \
    --env-runners 7 --checkpoint-every 10 --keep-checkpoints 3 --output runs/dec
```

The command only loads the file, applies the options it was given (`--controller`, `--regime`,
`--communication`, `--arrival-rate`, `--horizon`, `--budget-env-steps`, `--seed`,
`--checkpoint-every`, `--keep-checkpoints`) and hands the result to the trainer. It writes the
effective settings to `<output>/<name>_spec.json`, prints the same `Checkpoint saved to <path>`
line as example 04 and then the best checkpoint. Example 04 reads its hyperparameters, model
sizes and checkpointing from the same file, and keeps its own copy of the instance and
experiment values, which `tests/unit/test_training_spec.py` checks against the file.

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

## Comparing the conditions of a study

One run folder holds one condition per arm. `viz/study_report.py` puts several folders side by
side, so the arms and a variant such as the decentralised arm with communication (run into a
folder of its own) share the same panels. `examples/05_full_comparison.py` adds a load sweep, the
checkpoints and the reference planners evaluated at several arrival rates on the same seeds.

```bash
python examples/05_full_comparison.py --out sweep --planner token_passing \
    --checkpoint centralised=<path> --checkpoint section=<path> \
    --checkpoint decentralised=<path> --checkpoint decentralised+comm=<path> \
    --arrival-rates 0.15 0.25 0.3 0.4
python -m slap_mapd_coupling.viz.study_report --run runs/base --run comm=runs/comm \
    --sweep sweep --out report
```

The report writes `study_overview.png` (training return, tasks completed and overridden moves
over the iterations, the reward terms, and the evaluation episodes, one line or dot per
condition), `load_sweep.png` (completion share, service time and backlog against the arrival
rate, with the range over seeds) and `study_summary.md` with the numbers and their sample sizes.
It reads values off the files and says nothing about whether two conditions differ, because one
checkpoint per arm gives no spread over training runs. A checkpoint trained with
`--communication` is named `decentralised+comm` in the sweep and must be evaluated with the flag,
which the sweep checks against the module.

## Running a whole campaign unattended

`scripts/run_campaign.py` trains every arm at every load level, evaluates the checkpoints and
the reference planner on the same seeds, and writes the report, in one command. It keeps going
when a run fails, and the same command started again skips the runs that are done.

```bash
python scripts/run_campaign.py --out ~/thesis_runs/campaign --env-runners 7 --gpu
python scripts/run_campaign.py --out /tmp/plan --dry-run   # the plan only
```

It refuses to start on a checkout without the wait cost in the reward and the keep-up threshold.
Each run's output moves from `examples/output` into `OUT/runs/<condition>_rate<rate>/`, so do not
start other examples while it runs. `OUT/CAMPAIGN.md` lists what ran and what failed, and
`OUT/report/` holds one overview per load level and a load sweep across levels.

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
