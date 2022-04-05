from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import torch
from .named_dataset import NamedDataset
from torch.utils.data import Dataset
from torch import nn

class LookaheadSequenceDataset(Dataset, NamedDataset):
    name = "torch_lookahead"

    def __init__(
        self,
        seqs: List[Dict[str, np.ndarray]],
        x_features: List[str],
        y_features: List[str],
        delay_steps: int = 0,
        n_steps: int = 1,
    ) -> None:
        super().__init__()

        x, y, t = self.__class__._rollout_sequences(seqs, x_features, y_features, delay_steps, n_steps)

        # Data for training
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.t = torch.from_numpy(t)

        # Data for visualization
        # Take only first sequence
        seq = seqs[0]
        self.first_seq_x = np.concatenate([
            seq[f] for f in x_features
        ], axis=1)
        self.first_seq_y = np.concatenate([
            seq[f] for f in y_features
        ], axis=1)
        if "time" in seq:
            self.first_seq_t = seq["time"]
        else:
            self.first_seq_t = np.ones_like(self.first_seq_x)


        self.delay_steps = delay_steps
        self.n_steps = n_steps

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.t[index]

    @staticmethod
    def _relative_pose(query_pose: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
        diff = query_pose - reference_pose

        # Make a vector to be used for projection along theta
        proj_along_theta_vec = np.vstack((np.ones(len(reference_pose)), np.tan(reference_pose[:, 2]))).T
        assert proj_along_theta_vec.shape == (len(reference_pose), 2)
        proj_along_theta_vec = (proj_along_theta_vec.T / np.linalg.norm(proj_along_theta_vec, axis=1)).T

        # Make a projection vector orthogonal to one along theta
        # Specifically this one is rotated 90 degrees to the left
        proj_ortho_theta_vec = proj_along_theta_vec[:, [1, 0]]
        proj_ortho_theta_vec[:, 0] *= -1

        # Actually calculate projection of displacement onto both along and ortho vectors.
        # Then take inner product to get x (along) and y (values).
        disp_x = np.sum((np.sum(diff[:, :2] * proj_along_theta_vec, axis=1) * proj_along_theta_vec.T).T * diff[:, :2], axis=1)
        disp_y = np.sum((np.sum(diff[:, :2] * proj_ortho_theta_vec, axis=1) * proj_ortho_theta_vec.T).T * diff[:, :2], axis=1)


        result = np.vstack((disp_x, disp_y, diff[:, 2])).T

        return result

    @staticmethod
    def _rollout_sequences(
        seqs: List[Dict[str, np.ndarray]],
        x_features: List[str],
        y_features: List[str],
        delay_steps: int,
        n_steps: int = 1,
        remove_duplicates: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Pre-allocate arrays, get indexes of corresponding features
        seqs_len = 0
        for s in seqs:
            seqs_len += len(s[x_features[0]]) - n_steps - delay_steps

        if remove_duplicates:
            x_seqs = []
            y_seqs = []
            t_seqs = []
        else:
            x_seqs = None
            y_seqs = None
            t_seqs = np.zeros((seqs_len, 1), dtype=np.float32)

        # Process data
        seqs_so_far = 0
        for s in seqs:
            # Get data for sequence
            s_len = len(s[x_features[0]]) - n_steps - delay_steps
            x_s = np.concatenate([
                s[f][:s_len] for f in x_features
            ], axis=1)
            y_s = np.concatenate([
                s[f][-s_len:] for f in y_features
            ], axis=1)

            if "time" in s:
                t_s = s["time"][n_steps + delay_steps:] - s["time"][:s_len]
                t_s = t_s[:, None]
            else:
                t_s = np.ones((s_len, 1))

            if remove_duplicates:
                # Check for differences being almost 0
                idxs = np.logical_not(np.all(np.isclose(y_s, 0), axis=1))

                y_s = y_s[idxs]
                t_s = t_s[idxs]
                x_s = x_s[idxs]

                # Append to result
                x_seqs.extend(x_s)
                y_seqs.extend(y_s)
                t_seqs.extend(t_s)
            else:
                # If it's a first sequence define x_seqs and y_seqs with proper shapes
                if x_seqs is None:
                    x_seqs = np.zeros((seqs_len, x_s.shape[1]), dtype=np.float32)
                    y_seqs = np.zeros((seqs_len, y_s.shape[1]), dtype=np.float32)

                # Append to result
                x_seqs[seqs_so_far:seqs_so_far + s_len] = x_s
                y_seqs[seqs_so_far:seqs_so_far + s_len] = y_s
                t_seqs[seqs_so_far:seqs_so_far + s_len] = t_s
                seqs_so_far += s_len

        if remove_duplicates:
            x_seqs = np.array(x_seqs, dtype=np.float32)
            y_seqs = np.array(y_seqs, dtype=np.float32)
            t_seqs = np.array(t_seqs, dtype=np.float32)

        return x_seqs, y_seqs, t_seqs

    def rollout_single_sequence(self, model: Optional[nn.Module] = None, start_idx: int = 0, delay_steps: int = None):
        if delay_steps is None:
            delay_steps = self.delay_steps
        assert delay_steps >= 0, "Delay steps has to be greater or equal to 0"
        if start_idx < delay_steps:
            warnings.warn("delay_steps is less than start_idx, setting start_idx to be equal to delay_steps")
            start_idx = delay_steps

        start_state = self.first_seq_y[start_idx]

        seq = np.zeros((n_steps+1, one_steps.shape[1]))
        seq[0] = start_state
        for t in range(len(self.first_seq_x) - start_idx):
            idx = t + start_idx - delay_steps
            cur_angle = seq[idx, 2]

            if model is not None:
                with torch.no_grad():
                    summand = model(self.x[t][None, ...])
            else:
                summand = self.y

            assert(summand.shape == (3,))
            world_summand = np.zeros_like(summand)
            world_summand[0] = np.cos(cur_angle) * summand[0] - np.sin(cur_angle) * summand[1]
            world_summand[1] = np.sin(cur_angle) * summand[0] + np.cos(cur_angle) * summand[1]
            world_summand[2] = summand[2]
            seq[t+1, :] = seq[t, :] + world_summand

        return seq

    @staticmethod
    def rollout_single_seq(seq: np.ndarray):
        result = np.zeros((len(seq) + 1, seq.shape[1]))
        for t in range(len(seq)):
            cur_angle = seq[t, 2]

            summand = seq[t]

            assert(summand.shape == (3,))
            world_summand = np.zeros_like(summand)
            world_summand[0] = np.cos(cur_angle) * summand[0] - np.sin(cur_angle) * summand[1]
            world_summand[1] = np.sin(cur_angle) * summand[0] + np.cos(cur_angle) * summand[1]
            world_summand[2] = summand[2]
            result[t+1, :] = seq[t, :] + world_summand

        return result
