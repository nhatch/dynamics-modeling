from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn

class LookaheadDataset(Dataset):
    name = "torch_lookahead"

    def __init__(
        self,
        seqs: List[Dict[str, np.ndarray]],
        x_features: List[str],
        y_features: List[str],
        delay_steps: int = 1,
    ) -> None:
        super().__init__()

        x, y, t = self.__class__._rollout_sequences(seqs, x_features, y_features, delay_steps)

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

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.t[index]

    @staticmethod
    def _rollout_sequences(
        seqs: List[Dict[str, np.ndarray]],
        x_features: List[str],
        y_features: List[str],
        delay_steps: int,
        remove_duplicates: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Pre-allocate arrays, get indexes of corresponding features
        seqs_len = 0
        for s in seqs:
            seqs_len += max(0, len(s[x_features[0]]) - delay_steps)

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
            s_len = len(s[x_features[0]]) - delay_steps
            if s_len <= 0:
                # Sequence too short to process
                continue
            x_s = np.concatenate([
                s[f][:s_len] for f in x_features
            ], axis=1)
            y_s = np.concatenate([
                s[f][-s_len:] for f in y_features
            ], axis=1)

            if "time" in s:
                t_s = s["time"][delay_steps:] - s["time"][:-delay_steps]
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
