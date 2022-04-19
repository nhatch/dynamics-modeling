from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LookaheadDiffDataset(Dataset):
    name = "torch_lookahead_diff"

    def __init__(
        self,
        seqs: List[Dict[str, np.ndarray]],
        x_features: List[str],
        y_features: List[str],
        delay_steps: int = 0,
        n_steps: int = 1,
    ) -> None:
        super().__init__()

        x, y, t = self.__class__._rollout_sequences(
            seqs, x_features, y_features, delay_steps, n_steps
        )

        # Data for training
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.t = torch.from_numpy(t)

        # Data for visualization
        # Take only first sequence
        seq = seqs[0]
        self.first_seq_x = np.concatenate([seq[f] for f in x_features], axis=1)
        self.first_seq_y = np.concatenate([seq[f] for f in y_features], axis=1)
        if "time" in seq:
            self.first_seq_t = seq["time"]
        else:
            self.first_seq_t = np.ones_like(self.first_seq_x)

        self.delay_steps = delay_steps
        self.n_steps = n_steps

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.t[index]

    @staticmethod
    def _relative_pose(
        query_pose: np.ndarray, reference_pose: np.ndarray
    ) -> np.ndarray:
        diff = query_pose - reference_pose

        # Make a vector to be used for projection along theta
        proj_along_theta_vec = np.vstack(
            (np.ones(len(reference_pose)), np.tan(reference_pose[:, 2]))
        ).T
        assert proj_along_theta_vec.shape == (len(reference_pose), 2)
        proj_along_theta_vec = (
            proj_along_theta_vec.T / np.linalg.norm(proj_along_theta_vec, axis=1)
        ).T

        # Make a projection vector orthogonal to one along theta
        # Specifically this one is rotated 90 degrees to the left
        proj_ortho_theta_vec = proj_along_theta_vec[:, [1, 0]]
        proj_ortho_theta_vec[:, 0] *= -1

        # Actually calculate projection of displacement onto both along and ortho vectors.
        # Then take inner product to get x (along) and y (values).
        disp_x = np.sum(
            (
                np.sum(diff[:, :2] * proj_along_theta_vec, axis=1)
                * proj_along_theta_vec.T
            ).T
            * diff[:, :2],
            axis=1,
        )
        disp_y = np.sum(
            (
                np.sum(diff[:, :2] * proj_ortho_theta_vec, axis=1)
                * proj_ortho_theta_vec.T
            ).T
            * diff[:, :2],
            axis=1,
        )

        result = np.vstack((disp_x, disp_y, diff[:, 2])).T

        return result

    @staticmethod
    def rollout_single_seq(seq: np.ndarray):
        result = np.zeros((len(seq) + 1, seq.shape[1]))
        for t in range(len(seq)):
            cur_angle = seq[t, 2]

            summand = seq[t]

            assert summand.shape == (3,)
            world_summand = np.zeros_like(summand)
            world_summand[0] = (
                np.cos(cur_angle) * summand[0] - np.sin(cur_angle) * summand[1]
            )
            world_summand[1] = (
                np.sin(cur_angle) * summand[0] + np.cos(cur_angle) * summand[1]
            )
            world_summand[2] = summand[2]
            result[t + 1, :] = seq[t, :] + world_summand

        return result
