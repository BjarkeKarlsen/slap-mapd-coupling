## New Model Integration

To add a new model (e.g., `EnhancedModel`) for the decentralised controller:

1. Create `src/slap_mapd_coupling/models/enhanced_model.py`
2. Inherit from `BaseModel` or `TorchModelV2`
3. Register in `src/slap_mapd_coupling/models/registry.py`
4. Add a unit test under `tests/unit/`
5. Add/point an `examples/04_decentralised_training.py` variant at it

Usage:
```python
from slap_mapd_coupling.models import EnhancedModel
from slap_mapd_coupling.training.config import build_config

config = build_config(model_class=EnhancedModel)
trainer.train(config)
```
