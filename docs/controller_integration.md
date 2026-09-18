## New Controller Integration

To add a new controller architecture (a new realisation of `pi^route`,
Table `tab:information`):

1. Create `src/slap_mapd_coupling/controllers/<name>.py`, implementing the
   `Controller` protocol in `controllers/base.py`. Task assignment
   (`controllers/assignment.py`, `eq:assignmentrule`) is shared and does
   not need to be reimplemented - only routing differs.
2. Register it in `src/slap_mapd_coupling/controllers/registry.py`
3. Add a unit test under `tests/unit/`
4. Add/point an `examples/03_section_based.py`-style run at it

Because `environment/multi_agent_env.py` only ever calls the active
controller through the registry, and the conflict-resolution operator
(`resolution/conflict_resolution.py`) is applied identically regardless of
which controller proposed the joint action, adding a controller never
touches the environment loop, storage rules, or evaluation.

Usage:
```python
from slap_mapd_coupling.controllers.registry import get_controller

controller = get_controller("section")
actions = controller.route(observations)
```
