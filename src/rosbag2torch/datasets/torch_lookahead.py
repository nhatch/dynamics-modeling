from bisect import bisect_right
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .__defs import RawSequences


class LookaheadDataset(Dataset):
    def __init__(
        self,
        seqs: List[Dict[str, np.ndarray]],
        features: List[str],
        delayed_features: List[str],
        delay_steps: int = 1,
    ) -> None:
        super().__init__()

        self.processed_sequences, sequences_lengths = self.__parse_sequences(
            seqs, features, delayed_features, delay_steps
        )

        self.__features = features
        self.__delayed_features = delayed_features

        self.__longest_sequence_idx = np.argmax(sequences_lengths)
        self.__longest_sequence_length = sequences_lengths[self.__longest_sequence_idx]
        self.__total_len = sum(sequences_lengths)

        self.__sequence_start_idxs = np.cumsum(sequences_lengths) - sequences_lengths[0]

        self.__delay_steps = delay_steps

    def __len__(self) -> int:
        return self.__total_len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        sequence_idx = bisect_right(self.__sequence_start_idxs, index) - 1
        item_idx = index - self.__sequence_start_idxs[sequence_idx]

        result = []
        for f in self.__features:
            result.append(self.processed_sequences[sequence_idx][f][item_idx])
        for f in self.__delayed_features:
            result.append(
                self.processed_sequences[sequence_idx][f"{f}_delayed"][item_idx]
            )
        result.append(self.processed_sequences[sequence_idx]["time"][item_idx])
        return tuple(result)

    @property
    def longest_rollout(self) -> Tuple[torch.Tensor, ...]:
        start_idx = self.__sequence_start_idxs[self.__longest_sequence_idx]
        end_idx = start_idx + self.__longest_sequence_length
        feature_lists = None
        for idx in range(start_idx, end_idx, self.__delay_steps):
            features = self[idx]
            if feature_lists is None:
                feature_lists = [[f_tensor] for f_tensor in features]
            else:
                for f_idx, f_tensor in enumerate(features):
                    feature_lists[f_idx].append(f_tensor)

        if feature_lists is None:
            return tuple()

        return tuple(torch.tensor(f_list) for f_list in feature_lists)

    @staticmethod
    def __parse_sequences(
        sequences: RawSequences,
        features: List[str],
        delayed_features: List[str],
        delay_steps: int,
        *args,
        **kwargs,
    ):
        processed_sequences: List[Dict[str, torch.Tensor]] = []
        sequence_lengths: List[int] = []

        required_features = set(features) | set(delayed_features)

        for seq in sequences:
            if not required_features.issubset(seq.keys()):
                # Sequence does not contain all required features
                continue

            cur_seq = {}

            # Add torch sequences
            for f in features:
                cur_seq[f] = torch.from_numpy(seq[f][:-delay_steps])
            for f in delayed_features:
                cur_seq[f"{f}_delayed"] = torch.from_numpy(seq[f][delay_steps:])
            cur_seq["time"] = torch.from_numpy(
                seq["time"][delay_steps:] - seq["time"][:-delay_steps]
            )

            processed_sequences.append(cur_seq)

            cur_seq_len = len(cur_seq["time"])

            sequence_lengths.append(cur_seq_len)

        return processed_sequences, sequence_lengths
