import torch
from torch import nn

from ..abstract_model import ModelWithSpec


class Hidden2ReLU(nn.Module, ModelWithSpec):
    name = "hidden-2-relu"
    x_features = ["state", "control"]
    y_features = ["target"]

    dataset_name = "torch_lookahead"
    opt_algo = "regression_train_loop"

    # dim_in: 2 (state) + 3 (control)
    in_features = 5
    # dim_out: 3 (state displacement)
    out_features = 2

    def __init__(self) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(self.__class__.in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.__class__.out_features),
        )

    def forward(self, x):
        return self.layer(x)
