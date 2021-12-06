from typing import Optional
import torch
from torch.utils.data import Dataset
from torch import nn

class TorchModuleWrapper:
    def __init__(self, torch_model: nn.Module):
        self.model = torch_model

    def relative_pose(self, query_pose: torch.Tensor, reference_pose: torch.Tensor) -> torch.Tensor:
        pass

    def _pose(self, query_pose: torch.Tensor, reference_pose: torch.Tensor) -> torch.Tensor:
        pass

    def train(self, train_set: Dataset, epochs, val_set: Optional[Dataset])