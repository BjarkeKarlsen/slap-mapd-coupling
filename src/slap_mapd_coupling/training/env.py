"""RLlib MultiAgentEnv adapter around WarehouseMAPDEnv (issue #30).

environment/multi_agent_env.py's own module docstring names this
module's job explicitly: "Wiring [WarehouseMAPDEnv] to RLlib's actual
MultiAgentEnv base class is M5 scope (training/)" -- kept out of
environment/ so importing ray (a heavy RL dependency) is never required
for a baseline (centralised/section-based) run.

This adapter does NOT reimplement the six-stage loop: it constructs one
WarehouseMAPDEnv per episode and drives it entirely through the
`actions` parameter environment/multi_agent_env.py's step() gained for
this exact purpose (issue #30) -- so this file only translates between
RLlib's gym-style, action-driven MultiAgentEnv contract and
WarehouseMAPDEnv's own (full-state, controller-driven-OR-externally-
driven) contract:
  - observation: o_i(t) (#25's build_observation) encoded to a fixed
    shape (#28's encode_observation) plus its action mask, wrapped as
    {"action_mask": ..., "observations": ...} (#26's
    masked_observation_space contract);
  - action: RLlib hands back one action SLOT per agent (Discrete(d_max+1),
    eq:mask), passed straight through to WarehouseMAPDEnv.step(actions).

Only ever constructed with config.controller="decentralised" -- the
other two architectures already have their own env-driven loop
(examples/01, examples/03) and never need this adapter.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from slap_mapd_coupling.core.agents import AgentId
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.action_masking import masked_observation_space
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.observation import build_observation
from slap_mapd_coupling.environment.spaces import action_space, legality_mask, max_out_degree
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_observation,
    encoded_observation_space,
)


class WarehouseMAPDMultiAgentEnv(MultiAgentEnv):
    """One WarehouseMAPDEnv episode, exposed as an RLlib MultiAgentEnv
    for the decentralised regime's PPO training (sec:method:training).
    See module docstring."""

    def __init__(
        self,
        graph: WarehouseGraph,
        skus: Mapping[SkuId, SkuType],
        initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
        storage_capacities: Mapping[VertexId, float],
        config: ExperimentConfig,
        encoding_config: LocalSubgraphEncodingConfig,
    ) -> None:
        if config.controller != "decentralised":
            raise ValueError(
                f"WarehouseMAPDMultiAgentEnv is only meaningful for "
                f"controller='decentralised', got {config.controller!r} -- the other "
                "two architectures drive WarehouseMAPDEnv directly (examples/01, /03)."
            )
        super().__init__()
        self.graph = graph
        self._skus = skus
        self._initial_storage_counts = initial_storage_counts
        self._storage_capacities = storage_capacities
        self._config = config
        self._encoding_config = encoding_config
        self._d_max = max_out_degree(graph)

        self.agents = self.possible_agents = [str(a) for a in range(1, config.num_agents + 1)]
        base_observation_space = encoded_observation_space(encoding_config)
        per_agent_observation_space = masked_observation_space(
            base_observation_space, self._d_max + 1
        )
        per_agent_action_space = action_space(graph)
        self.observation_spaces = {aid: per_agent_observation_space for aid in self.agents}
        self.action_spaces = {aid: per_agent_action_space for aid in self.agents}

        self._env = WarehouseMAPDEnv(
            graph, skus, initial_storage_counts, storage_capacities, config
        )

    # RLlib's own MultiAgentEnv.reset()/step() signatures are already
    # loosely typed against gymnasium.core.Env's single-agent generics
    # (see e.g. environment/action_masking.py's own untyped-surface
    # comment for the same upstream rough edge) -- narrowing our own
    # dict[str, ...] returns against that isn't meaningful, hence the
    # override ignores below rather than fighting the upstream generics.
    def reset(  # type: ignore[override]
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        super().reset(seed=seed, options=options)
        self._env.reset(seed=seed)
        return self._encode_all_observations(), {str(aid): {} for aid in self.agents}

    def step(  # type: ignore[override]
        self, action_dict: Mapping[str, int]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        actions: dict[AgentId, int] = {int(aid): int(slot) for aid, slot in action_dict.items()}
        _, rewards, terminated, truncated, infos = self._env.step(actions)

        observations = self._encode_all_observations()
        str_rewards = {str(aid): value for aid, value in rewards.items()}
        str_terminated = {str(aid): value for aid, value in terminated.items() if aid != "__all__"}
        str_terminated["__all__"] = bool(terminated["__all__"])
        str_truncated = {str(aid): value for aid, value in truncated.items() if aid != "__all__"}
        str_truncated["__all__"] = bool(truncated["__all__"])
        str_infos = {str(aid): value for aid, value in infos.items() if aid != "__all__"}
        str_infos["__all__"] = {}

        return observations, str_rewards, str_terminated, str_truncated, str_infos

    def _encode_all_observations(self) -> dict[str, Any]:
        locations = self._env.fleet.locations()
        observations: dict[str, Any] = {}
        for agent_id, location in locations.items():
            observation = build_observation(
                self.graph, self._env.fleet, self._env.tasks, agent_id, self._env.t, self._config
            )
            mask = legality_mask(self.graph, location, d_max=self._d_max)
            encoded = encode_observation(
                self.graph, location, observation, mask, self._encoding_config
            )
            observations[str(agent_id)] = encoded.as_rllib_dict()
        return observations
