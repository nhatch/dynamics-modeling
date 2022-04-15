from typing import Dict, List, Optional, Tuple, Union
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import Dataset

RawSequence = Dict[str, np.ndarray]
RawSequences = List[RawSequence]

class SequenceLookaheadDataset(Dataset):
    def __init__(
        self,
        sequences: RawSequences,
        features: List[str],
        delayed_features: List[str],
        delay_steps: int = 1,
        sequence_length: int = 50,
    ) -> None:
        # Set all of the variables
        self.__features = features
        self.__delayed_features = delayed_features
        self.__delay_steps = delay_steps
        self.__sequence_length = sequence_length

        self.processed_sequences, self.__sequence_lengths, self.__max_len_rollout_idx, self.__max_len_rollout = self.__class__.__parse_sequences(
            sequences,
            features=features,
            delayed_features=delayed_features,
            delay_steps=delay_steps,
            sequence_length=sequence_length,
        )
        # Index of first rollout in each sequence
        self.__sequence_start_idxs = np.cumsum([0] + self.__sequence_lengths[:-1])
        # Total number of rollouts in all sequences
        self.__total_len = sum(self.__sequence_lengths)

    def __len__(self) -> int:
        return self.__total_len

    def __get_sequence_of_length(self, index: int, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        sequence_idx = bisect_right(self.__sequence_start_idxs, index) - 1
        rollout_idx = index - self.__sequence_start_idxs[sequence_idx]

        offset_idx = rollout_idx % self.__delay_steps
        start_idx = rollout_idx // self.__delay_steps

        result = []
        for f in self.__features:
            result.append(
                self.processed_sequences[sequence_idx][offset_idx][f][start_idx:start_idx + sequence_length]
            )
        for f in self.__delayed_features:
            result.append(
                self.processed_sequences[sequence_idx][offset_idx][f"{f}_delayed"][start_idx:start_idx + sequence_length]
            )
        result.append(
            self.processed_sequences[sequence_idx][offset_idx]["time"][start_idx:start_idx + sequence_length]
        )
        return tuple(result)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return self.__get_sequence_of_length(index, self.__sequence_length)


    @property
    def longest_rollout(self) -> Tuple[torch.Tensor, ...]:
        return self.__get_sequence_of_length(self.__sequence_start_idxs[self.__max_len_rollout_idx], self.__max_len_rollout)

    @staticmethod
    def __parse_sequences(sequences: RawSequences, features: List[str], delayed_features: List[str], delay_steps: int, sequence_length: int, *args, **kwargs):
        processed_sequences: List[Dict[int, Dict[str, torch.Tensor]]] = []
        sequence_lengths: List[int] = []
        max_len_rollout = 0
        max_len_rollout_idx = 0

        required_features = set(features) | set(delayed_features)

        seq_idx = 0

        for seq in sequences:
            if not required_features.issubset(seq.keys()):
                # Sequence does not contain all required features
                continue

            cur_seq_flat = {}

            # Add torch sequences
            for f in features:
                cur_seq_flat[f] = torch.from_numpy(seq[f][:-delay_steps])
            for f in delayed_features:
                cur_seq_flat[f"{f}_delayed"] = torch.from_numpy(seq[f][delay_steps:])
            cur_seq_flat["time"] = torch.from_numpy(seq["time"][delay_steps:] - seq["time"][:-delay_steps])

            # Calculate the number of sequences that can be read
            # In each step we are taking delay_steps steps forward (to get next element of the rollout)
            # for sequence_length steps
            # One rollout will be (sequence_length - 1) * delay_steps long
            num_rollouts = len(cur_seq_flat["time"]) - (sequence_length - 1) * delay_steps
            # Example: if we have sequence_length=3, delay_steps=2 and a sequence of length 10,
            # we can read sequences:
            # - [0, 2, 4]
            # - [1, 3, 5]
            # - [2, 4, 6]
            # - [3, 5, 7]
            # - [4, 6, 8]
            # - [5, 7, 9]

            # Sequence too short. No rollouts can be read
            if num_rollouts <= 0:
                continue

            # From the example above you can see that there are at most delay_steps independent sequences
            # defined by their starting index.
            # - [0, 2, 4, 6, 8]
            # - [1, 3, 5, 7, 9]
            cur_seq = {}
            for start_idx in range(min(delay_steps, num_rollouts)):
                cur_seq[start_idx] = {}
                for f, val in cur_seq_flat.items():
                    cur_seq[start_idx][f] = val[start_idx::delay_steps].contiguous()

            # Add sequence to processed sequences
            processed_sequences.append(cur_seq)

            # Add it's length to the sequence lengths
            sequence_lengths.append(num_rollouts)

            # Determine whether it is the longest sequence
            if max_len_rollout < len(cur_seq[0]["time"]):
                max_len_rollout = len(cur_seq[0]["time"])
                max_len_rollout_idx = seq_idx

            # Only update seq_idx if we have a valid sequence
            seq_idx += 1

        return processed_sequences, sequence_lengths, max_len_rollout_idx, max_len_rollout