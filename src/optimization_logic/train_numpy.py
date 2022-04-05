from calendar import day_abbr
from typing import Type
from data_utils.datasets.numpy_set import NumpyDataset
from models.numpy_models.abstract_model import AbstractNumpyModel
from models.numpy_models.models import LinearModel


def train(
    dataset: NumpyDataset,
    model: Type[AbstractNumpyModel],
    delay_steps: int = 1,
):
    # Initilize model
    model = model(dataset.D, dataset.H, delay_steps)
    model.train(dataset.dataset)
    return model
