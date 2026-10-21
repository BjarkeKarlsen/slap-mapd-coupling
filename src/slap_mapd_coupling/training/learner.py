"""The PPO learner of every learned arm (sec:method:training).

It is RLlib's PPOTorchLearner with two additions to how a train batch reaches the module.
Neither changes the loss, the minibatches or the updates.

Context rows for messages. With communication on, a compact row names each sender by the
key of the sender's own row (models/window_tables.py). PPO cuts the train batch into
minibatches by row, so a sender's row can be outside the minibatch of a robot that hears it.
For each minibatch the learner looks those rows up in the whole train batch and adds them
to the minibatch's observation as context rows (`CONTEXT_KEYS`). The module embeds them with
the current weights and reads the messages from them, but they carry no loss. A minibatch
therefore computes exactly what the same rows give with their senders' views stored in the
row, and the receiver's loss reaches the sender's view through the shared weights.

The batch in host memory (`batch_on_cpu`). RLlib builds the train batch on the learner's
device, so a whole iteration must fit on the GPU. With `batch_on_cpu` the batch is built in
host memory instead, the value pass moves one chunk at a time to the GPU
(environment/action_masking.py), and each minibatch is moved to the GPU before its update.
It costs a copy per minibatch and is off by default. The setting arrives through
`learner_config_dict`, which training/config.py fills.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.core.columns import Columns

BATCH_ON_CPU = "batch_on_cpu"

# The keys of a compact row that a context row copies (models/window_tables.py), and the
# context key each one becomes.
_CONTEXT_SOURCE = {
    "context_row_key": "row_key",
    "context_vertex": "vertex",
    "context_eta": "eta",
    "context_flags": "flags",
}


def add_context_rows(
    minibatch_obs: dict[str, torch.Tensor],
    full_obs: dict[str, torch.Tensor],
    sorted_keys: torch.Tensor,
    order: torch.Tensor,
) -> None:
    """Adds to `minibatch_obs` the rows of `full_obs` that its senders need and
    it does not hold. `sorted_keys` and `order` are torch.sort of the full
    batch's row keys. The context rows are distinct and on the minibatch's
    device. Raises ValueError if a sender's row is not in the full batch."""
    senders = minibatch_obs["sender_row_key"].reshape(-1)
    wanted = torch.unique(senders[senders >= 0])
    own = minibatch_obs["row_key"]
    wanted = wanted[~torch.isin(wanted, own.to(wanted.device))]
    position = torch.searchsorted(sorted_keys, wanted.to(sorted_keys.device))
    position = position.clamp(max=max(len(sorted_keys) - 1, 0))
    if len(wanted) and not bool((sorted_keys[position] == wanted.to(sorted_keys.device)).all()):
        raise ValueError("a sender's row is not in the train batch, so its message is missing.")
    rows = order[position]
    device = own.device
    for context_key, source in _CONTEXT_SOURCE.items():
        minibatch_obs[context_key] = full_obs[source][rows.to(full_obs[source].device)].to(device)


class GNNPPOTorchLearner(PPOTorchLearner):
    """PPOTorchLearner with context rows for messages and an optional train batch in host
    memory, see the module docstring."""

    def build(self, **kwargs: Any) -> None:
        super().build(**kwargs)
        if self._batch_on_cpu() and self._learner_connector is not None:
            # Keep RLlib's pipeline, which PPOLearner.build extends with the advantage
            # estimation, and only point its tensor conversion at host memory.
            converters = [
                piece
                for piece in self._learner_connector.connectors
                if isinstance(piece, NumpyToTensor)
            ]
            if not converters:
                raise RuntimeError("the learner connector has no NumpyToTensor to move to the CPU.")
            for piece in converters:
                piece._device = torch.device("cpu")

    def _batch_on_cpu(self) -> bool:
        return bool(self.config.learner_config_dict.get(BATCH_ON_CPU, False))

    def _make_batch_if_necessary(self, training_data):  # type: ignore[no-untyped-def]
        batch = super()._make_batch_if_necessary(training_data)
        # Index each module's row keys once per update, for the context rows.
        self._row_index: dict[str, tuple[dict, torch.Tensor, torch.Tensor]] = {}
        for module_id, module_batch in batch.policy_batches.items():
            obs = module_batch[Columns.OBS]
            if isinstance(obs, dict) and "sender_row_key" in obs:
                sorted_keys, order = torch.sort(obs["row_key"])
                self._row_index[module_id] = (obs, sorted_keys, order)
        return batch

    def _create_iterator_if_necessary(self, **kwargs: Any) -> Any:
        return self._prepared(super()._create_iterator_if_necessary(**kwargs))

    def _prepared(self, inner: Any) -> Iterator[Any]:
        device: Optional[torch.device] = self._device if self._batch_on_cpu() else None
        for minibatch in inner:
            for module_id, module_batch in minibatch.policy_batches.items():
                if device is not None:
                    module_batch.to_device(device, framework="torch")
                index = getattr(self, "_row_index", {}).get(module_id)
                if index is not None:
                    full_obs, sorted_keys, order = index
                    add_context_rows(module_batch[Columns.OBS], full_obs, sorted_keys, order)
            yield minibatch
