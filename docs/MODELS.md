# The model of a learned arm

Every learned arm uses one network, `models/rl_module.py::GNNActionMaskingRLModule`, built on
RLlib's new API stack. The arms differ in what a robot observes and who chooses together, not in
the network. `GNNEncoderConfig` and `PolicyValueHeadConfig` set its sizes.

## What a robot sees

- The window, a set of vertices. It is the field of view for the decentralised arm, every movable
  vertex for the centralised arm and the full-view setting, and the zone with its halo for the
  section arm. A window is never truncated, so `max_local_nodes` must fit it.
- Eight features per vertex (`NODE_FEATURE_DIM`): the multi-hot role (storage, delivery, endpoint,
  transit), the distance label, whether another robot stands there, and whether the vertex is
  claimed. Claimed means an earlier robot of the same decision has chosen to end the timestep on
  it. It is always 0 for the decentralised arm and the full-view setting. The eighth, last,
  marks a receiving vertex, where arriving stock enters.
- The action mask. It holds the legal moves from the robot's vertex. For the centralised and
  section arms it also holds the claim mask, which switches off a move onto a claimed vertex, onto
  a robot that has not chosen yet, or a swap with an earlier robot. Waiting is never switched off.

## Readout and critic

The wider arms (centralised, section, full-view) pool the window into one vector (`pooling="mean"`).
The decentralised arm does not (`pooling="none"`). The critic reads the same readout as the
policy, so it sees the claims too.

## Turns

A timestep is one or more turns, and the module is called once per turn for the robots that
choose in it (`training/policy.py::ModulePolicy`). Each robot's choice is one training sample with
its own value estimate. See `docs/TRAINING.md` for the credit and the budget.

## Adding a model

See `docs/model_integration.md`.
