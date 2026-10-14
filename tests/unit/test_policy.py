"""Unit tests for slap_mapd_coupling.training.policy, and run_episode with a learned arm."""

import json

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType, is_feasible
from slap_mapd_coupling.environment.zones import zones_by_aisle
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_instance
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.models.rl_module import GNNActionMaskingRLModule
from slap_mapd_coupling.training import policy as policy_module
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv
from slap_mapd_coupling.training.policy import ModulePolicy

_POOLING = {"decentralised": "none", "centralised": "mean", "section": "mean"}
ARMS = list(_POOLING)

# The largest field of view of the instance at d_obs = 3 has 20 vertices, and the instance has 30
# movable vertices, the centralised window and the largest section window. A window is never
# truncated, so the cap must fit it.
_CAP = {"decentralised": 20, "centralised": 30, "section": 30}


def _instance():
    params = GeneratorParams(
        max_num_aisles=6,
        max_aisle_length=6,
        max_num_cross_aisles=5,
        one_way_fraction=0.0,
        wait_cost=0.0,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=4,
        seed=0,
    )
    return generate_instance(params)


def _storage(graph):
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    return skus, capacities, counts


def _config(controller: str, **overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        controller=controller,
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=0.5,
        seed=7,
        horizon=12,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class _Arm:
    """One arm on the instance, with everything run_episode and the adapter need."""

    def __init__(self, controller: str, communication: bool = False, **config_overrides):
        instance = _instance()
        self.graph = instance.graph
        self.skus, self.capacities, self.counts = _storage(self.graph)
        self.config = _config(controller, communication=communication, **config_overrides)
        self.encoding_config = LocalSubgraphEncodingConfig(
            max_local_nodes=_CAP[controller], max_messages=2
        )
        self.pooling = _POOLING[controller]
        self.zones = (
            zones_by_aisle(instance.graph, instance.aisle_index, 2)
            if controller == "section"
            else None
        )

    def adapter(self) -> WarehouseMAPDMultiAgentEnv:
        return WarehouseMAPDMultiAgentEnv(
            self.graph,
            self.skus,
            self.counts,
            self.capacities,
            self.config,
            self.encoding_config,
            (self.config.seed,),
            credit_signal="per_agent",
            pooling=self.pooling,
            zones=self.zones,
        )

    def module(self):
        """An untrained module for the spaces of this arm's adapter."""
        env = self.adapter()
        spec = RLModuleSpec(
            module_class=GNNActionMaskingRLModule,
            observation_space=env.observation_spaces["1"],
            action_space=env.action_spaces["1"],
            model_config={
                "gnn_encoder_config": {
                    "num_rounds": 1,
                    "hidden_width": 8,
                    "communication": self.config.communication,
                    "pooling": self.pooling,
                },
                "policy_value_head_config": {"hidden_width": 8},
            },
        )
        return spec.build()

    def run_episode(self, module, trace_path=None):
        policy = ModulePolicy(module)
        return run_episode(
            self.graph,
            self.skus,
            self.counts,
            self.capacities,
            self.config,
            trace_path,
            policy=policy,
            encoding_config=self.encoding_config,
            zones=self.zones,
            pooling=self.pooling,
        )


class _CountingModule:
    """Wraps an RLModule and records the batch size of every forward_inference call."""

    def __init__(self, module):
        self._module = module
        self.batch_sizes: list[int] = []

    def forward_inference(self, batch):
        self.batch_sizes.append(batch[Columns.OBS]["action_mask"].shape[0])
        return self._module.forward_inference(batch)


def _per_agent_slots(module, observations, t, episode_seed):
    """The slots of one timestep from one module call per agent, drawn from the
    same (episode_seed + offset, t, agent_id) streams as ModulePolicy."""
    offset = policy_module._ACTION_SAMPLING_SEED_OFFSET
    slots = {}
    for agent_id, observation in observations.items():
        out = module.forward_inference(policy_module._batch([observation]))
        logits = out[Columns.ACTION_DIST_INPUTS][0].detach().numpy()
        rng = np.random.default_rng((episode_seed + offset, t, int(agent_id)))
        slots[agent_id] = policy_module._sample_legal_slot(logits, observation["action_mask"], rng)
    return slots


@pytest.mark.parametrize("communication", [False, True])
def test_one_module_call_per_step_for_all_robots(communication):
    arm = _Arm("decentralised", communication)
    counting = _CountingModule(arm.module())
    policy = ModulePolicy(counting)
    env = arm.adapter()
    observations, _ = env.reset(seed=3)
    policy.reset(3)

    for _ in range(4):
        observations = env.step(policy(observations, env.core.t))[0]

    assert counting.batch_sizes == [3, 3, 3, 3]


@pytest.mark.parametrize(
    "controller, communication",
    [("decentralised", False), ("decentralised", True), ("centralised", False), ("section", False)],
)
def test_actions_equal_those_from_one_module_call_per_agent(controller, communication):
    arm = _Arm(controller, communication)
    module = arm.module()
    policy = ModulePolicy(module)
    env = arm.adapter()
    observations, _ = env.reset(seed=3)
    policy.reset(3)

    for _ in range(8):
        t = env.core.t
        expected = _per_agent_slots(module, observations, t, 3)
        slots = policy(observations, t)
        assert slots == expected
        observations = env.step(slots)[0]


@pytest.mark.parametrize("controller", ARMS)
def test_actions_are_always_legal(controller):
    arm = _Arm(controller)
    policy = ModulePolicy(arm.module())
    env = arm.adapter()
    observations, _ = env.reset(seed=5)
    policy.reset(5)

    for _ in range(arm.config.horizon):
        slots = policy(observations, env.core.t)
        assert set(slots) == set(observations)
        for agent_id, slot in slots.items():
            assert observations[agent_id]["action_mask"][slot] == 1.0
        observations = env.step(slots)[0]


def test_actions_are_reproducible_for_a_fixed_seed_t_and_agent():
    """The same episode seed replays bit for bit, however many episodes the same
    policy has already run through reset()."""
    arm = _Arm("decentralised")
    module = arm.module()
    env = arm.adapter()
    observations, _ = env.reset(seed=42)
    first, second = ModulePolicy(module), ModulePolicy(module)
    first.reset(42)
    second.reset(42)

    for t in (0, 1, 7):
        assert first(observations, t) == second(observations, t)

    # Re-resetting the same instance to an unrelated episode and back replays too, since
    # the stream comes from reset()'s argument and not from how many episodes came before.
    first.reset(999)
    first(observations, 0)
    first.reset(42)
    assert first(observations, 0) == second(observations, 0)


def test_different_episode_seeds_sample_different_slots_from_the_same_logits():
    arm = _Arm("decentralised")
    env = arm.adapter()
    observations, _ = env.reset(seed=0)
    policy = ModulePolicy(arm.module())

    seen = set()
    for seed in range(10):
        policy.reset(seed)
        seen.add(tuple(sorted(policy(observations, 0).items())))
    assert len(seen) > 1  # the difference can only come from the stream reset() seeds


def test_calling_before_reset_raises():
    arm = _Arm("decentralised")
    env = arm.adapter()
    observations, _ = env.reset(seed=0)
    policy = ModulePolicy(arm.module())

    with pytest.raises(RuntimeError, match="reset"):
        policy(observations, 0)


def test_a_call_with_no_robots_returns_no_slots():
    policy = ModulePolicy(_Arm("decentralised").module())
    policy.reset(0)
    assert policy({}, 0) == {}


def test_the_call_time_is_recorded_once_per_call_through_reset():
    arm = _Arm("decentralised")
    env = arm.adapter()
    observations, _ = env.reset(seed=0)
    policy = ModulePolicy(arm.module())
    recorded: list[float] = []

    policy.reset(0)
    policy(observations, 0)
    assert recorded == []

    policy.reset(0, record_runtime=recorded.append)
    policy(observations, 0)
    policy(observations, 1)
    assert len(recorded) == 2
    assert all(seconds > 0.0 for seconds in recorded)


def test_sample_legal_slot_never_picks_a_masked_out_slot():
    rng = np.random.default_rng(0)
    logits = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)  # would favour slots freely
    mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(200):
        slot = policy_module._sample_legal_slot(logits, mask, rng)
        assert mask[slot] == 1.0


@pytest.mark.parametrize("communication", [False, True])
def test_batched_logits_equal_the_per_agent_logits(communication):
    arm = _Arm("decentralised", communication)
    module = arm.module()
    env = arm.adapter()
    observations, _ = env.reset(seed=0)

    with torch.no_grad():
        together = module.forward_inference(policy_module._batch(list(observations.values())))
        for row, observation in enumerate(observations.values()):
            alone = module.forward_inference(policy_module._batch([observation]))
            assert torch.allclose(
                together[Columns.ACTION_DIST_INPUTS][row],
                alone[Columns.ACTION_DIST_INPUTS][0],
                atol=1e-5,
            )


@pytest.mark.parametrize("communication", [False, True])
def test_batch_stacks_every_observation_key_including_the_move_slot_arrays(communication):
    arm = _Arm("decentralised", communication)
    env = arm.adapter()
    observations, _ = env.reset(seed=0)
    d_max = env.action_spaces["1"].n - 1

    batch = policy_module._batch(list(observations.values()))

    stacked = batch[Columns.OBS]["observations"]
    assert set(stacked) == set(observations["1"]["observations"])
    assert stacked["move_target_index"].shape == (3, d_max)
    assert stacked["move_target_index"].dtype == torch.int64
    assert ("move_sender_index" in stacked) == communication
    assert ("move_sender_mask" in stacked) == communication


def test_batch_stacks_the_pool_mask_like_every_other_key():
    arm = _Arm("centralised")
    env = arm.adapter()
    observations, _ = env.reset(seed=0)

    stacked = policy_module._batch(list(observations.values()))[Columns.OBS]["observations"]

    expected = np.stack([o["observations"]["pool_mask"] for o in observations.values()])
    assert stacked["pool_mask"].dtype == torch.float32
    np.testing.assert_array_equal(stacked["pool_mask"].numpy(), expected)


@pytest.mark.parametrize("controller", ARMS)
def test_a_learned_episode_through_the_adapter_respects_the_correctness_invariants(controller):
    arm = _Arm(controller, horizon=20)
    policy = ModulePolicy(arm.module())
    env = arm.adapter()
    observations, _ = env.reset(seed=arm.config.seed)
    policy.reset(arm.config.seed)
    before = env.core.fleet

    for t in range(arm.config.horizon):
        observations = env.step(policy(observations, t))[0]
        now = t + 1
        assert is_collision_free(before, env.core.fleet), f"collision at t={t}"
        assert is_feasible(env.core.storage), f"infeasible storage at t={t}"
        for task in env.core.tasks:
            if task.release_time <= now:
                status = task.status(now)
                if task.finish_time is not None and now >= task.finish_time:
                    assert status == "completed"
                elif task.assignment_time is not None and now >= task.assignment_time:
                    assert status == "active"
                else:
                    assert status == "waiting"
        before = env.core.fleet


@pytest.mark.parametrize("controller", ARMS)
def test_the_routing_runtime_covers_every_decision_once_the_observation_build_and_the_policy(
    controller,
):
    arm = _Arm(controller)
    env = arm.adapter()
    policy = ModulePolicy(arm.module())
    observations, _ = env.reset(seed=arm.config.seed)
    policy.reset(arm.config.seed, record_runtime=env.core.log.record_routing_runtime)

    for _ in range(arm.config.horizon):
        observations = env.step(policy(observations, env.core.t))[0]

    # Per timestep, one observation build and one policy call. The observation after the
    # last step feeds no decision, so it is not recorded.
    assert len(env.core.log.routing_runtime_seconds) == 2 * arm.config.horizon


@pytest.mark.parametrize("controller", ARMS)
def test_run_episode_with_an_untrained_module(controller, tmp_path):
    arm = _Arm(controller, horizon=15)
    module = arm.module()
    path = tmp_path / "traces" / "episode.jsonl"

    first = arm.run_episode(module, path)

    assert first.controller == controller
    assert first.horizon == 15
    assert first.mean_decision_runtime_seconds > 0.0
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == arm.config.horizon + 1
    assert lines[0]["type"] == "header"
    _assert_no_conflicts_in_the_trace(lines)

    # Replay is bit for bit, and the trace does not change the metrics.
    second = arm.run_episode(module)
    exclude = {"mean_decision_runtime_seconds"}  # wall-clock time is not part of replay
    assert first.model_dump(exclude=exclude) == second.model_dump(exclude=exclude)


def _assert_no_conflicts_in_the_trace(lines):
    previous = {int(a): v for a, v in lines[0]["initial"].items()}
    for step in lines[1:]:
        now = {int(a): entry["at"] for a, entry in step["agents"].items()}
        assert len(set(now.values())) == len(now), f"vertex conflict at t={step['t']}"
        for a, b in ((a, b) for a in now for b in now if a < b):
            assert not (now[a] == previous[b] and now[b] == previous[a]), "swap conflict"
        previous = now


def test_run_episode_with_communication_replays_bit_for_bit():
    arm = _Arm("decentralised", communication=True, horizon=15)
    module = arm.module()

    first = arm.run_episode(module)
    second = arm.run_episode(module)

    exclude = {"mean_decision_runtime_seconds"}
    assert first.model_dump(exclude=exclude) == second.model_dump(exclude=exclude)


def test_run_episode_replays_with_a_policy_that_has_already_run_another_episode():
    """run_episode resets the policy to config.seed itself, so a policy left in the state of
    another episode gives the same result."""
    arm = _Arm("decentralised", horizon=15)
    module = arm.module()
    used = ModulePolicy(module)
    used.reset(999)

    def run(policy):
        return run_episode(
            arm.graph,
            arm.skus,
            arm.counts,
            arm.capacities,
            arm.config,
            policy=policy,
            encoding_config=arm.encoding_config,
            pooling=arm.pooling,
        )

    exclude = {"mean_decision_runtime_seconds"}
    assert run(used).model_dump(exclude=exclude) == run(ModulePolicy(module)).model_dump(
        exclude=exclude
    )


@pytest.mark.parametrize("controller", ARMS)
def test_run_episode_with_a_controller_config_and_no_policy_raises(controller):
    arm = _Arm(controller)
    with pytest.raises(ValueError, match="policy"):
        run_episode(arm.graph, arm.skus, arm.counts, arm.capacities, arm.config)


def test_run_episode_with_a_policy_but_no_encoding_config_or_pooling_raises():
    arm = _Arm("decentralised")
    policy = ModulePolicy(arm.module())
    with pytest.raises(ValueError, match="encoding_config"):
        run_episode(arm.graph, arm.skus, arm.counts, arm.capacities, arm.config, policy=policy)
    with pytest.raises(ValueError, match="pooling"):
        run_episode(
            arm.graph,
            arm.skus,
            arm.counts,
            arm.capacities,
            arm.config,
            policy=policy,
            encoding_config=arm.encoding_config,
        )
