# Model registration utility

from ray.rllib.models import ModelCatalog

from .base_model import BaseModel

ModelCatalog.register_custom_model("BaseModel", BaseModel)