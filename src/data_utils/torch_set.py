from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: List[np.ndarray],
        x_features: np.ndarray,
        y_features: np.ndarray,
        delay_steps: int = 1,
        n_steps: int = 1,
    ) -> None:
        super().__init__()

        x, y = self.__class__._rollout_sequences(seqs, x_features, y_features, delay_steps, n_steps)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    @staticmethod
    def _relative_pose(query_pose: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
        diff = query_pose - reference_pose
        distance = np.linalg.norm(diff[:, :2], axis=1)
        direction = np.arctan2(diff[:, 1], diff[:,0])
        relative_direction = direction - reference_pose[:, 2]
        angle_diff = diff[:, 2]
        minimized_angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        return np.array([
            distance*np.cos(relative_direction),
            distance*np.sin(relative_direction),
            minimized_angle_diff,
        ]).T

    @staticmethod
    def _rollout_sequences(
        seqs: List[np.ndarray],
        x_features: np.ndarray,
        y_features: np.ndarray,
        delay_steps: int,
        n_steps: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        seqs_len = 0
        # Pre-allocate arrays
        for s in seqs:
            seqs_len += len(s) - n_steps - delay_steps
        x_seqs = np.zeros((seqs_len, x_features.sum()), dtype=np.float32)
        y_seqs = np.zeros((seqs_len, y_features.sum()), dtype=np.float32)

        # Process data
        seqs_so_far = 0
        for s in seqs:
            # Get data for sequence
            s_len = len(s) - n_steps - delay_steps
            x_s = s[:s_len, x_features]
            y_s = s[:, y_features]

            # Get target y
            relative_targets = SequenceDataset._relative_pose(y_s[n_steps:], y_s[:-n_steps])
            y_s = relative_targets[delay_steps:]

            # Append to result
            x_seqs[seqs_so_far:seqs_so_far + s_len] = x_s
            y_seqs[seqs_so_far:seqs_so_far + s_len] = y_s
            seqs_so_far += s_len

        return x_seqs, y_seqs
