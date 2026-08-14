## New Model Integration

To add a new model (e.g., `EnhancedModel`):

1. Create `src/hierarchical_robot_rl/models/enhanced_model.py`
2. Inherit from `BaseModel` or `TorchModelV2`
3. Register in `src/hierarchical_robot_rl/models/registry.py`
4. Add unit test in `tests/unit/test_enhanced_model.py`
5. Add example in `examples/06_train_with_custom_model.py`
6. Update docs in `docs/MODELS.md`

Usage:
```python
from hierarchical_robot_rl.models import EnhancedModel
from hierarchical_robot_rl.training.config import build_config

config = build_config(model_class=EnhancedModel)
trainer.train(config)