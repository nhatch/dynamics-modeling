from functools import lru_cache
from typing import Dict, Type
from .abstract_model import ModelWithSpec
from inspect import isclass

@lru_cache(1)
def get_all_models() -> Dict[str, Type[ModelWithSpec]]:
    def filter_for_models_with_spec(tup):
        k, v = tup
        return (
            not k[0].startswith("__")  # __main__ etc.
            and isclass(v)
            and issubclass(v, ModelWithSpec)
        )

    from . import numpy_models, torch_models
    return (
        dict(map(
            lambda tup: (tup[1].name, tup[1]),
            filter(filter_for_models_with_spec , vars(numpy_models).items())
        ))
        | dict(map(
            lambda tup: (tup[1].name, tup[1]),
            filter(filter_for_models_with_spec , vars(torch_models).items())
        ))
    )

def load_model(model_type: str):
    models = get_all_models()
    if model_type not in models:
        raise ValueError(
            f"Model named {model_type} not found.\n"
            f"Available models are {', '.join(models.keys())}"
        )
    return models[model_type]
