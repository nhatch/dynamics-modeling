from bisect import bisect_right
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .__defs import RawSequences


class SequenceLookaheadDataset(Dataset):
    def __init__(
        self,
        sequences: RawSequences,
        features_with_delays: List[Tuple[str, int]],
        sequence_length: int = 50,
    ) -> None:

        # Sanitize features (i.e. make sure that minimum delay is 0)
        min_delay = min(delay for _, delay in features_with_delays)
        features_with_delays = [(f, delay - min_delay) for f, delay in features_with_delays]

        # Set all of the variables
        self.__features = features_with_delays
        self.__sequence_length = sequence_length

        (
            self.processed_sequences,
            self.__sequence_lengths,
            self.__max_len_rollout_idx,
            self.__max_len_rollout,
        ) = self.__class__.__parse_sequences(
            sequences,
            features=self.__features,
            sequence_length=sequence_length,
        )
        # Index of first rollout in each sequence
        self.__sequence_start_idxs = np.cumsum([0] + self.__sequence_lengths[:-1])
        # Total number of rollouts in all sequences
        self.__total_len = sum(self.__sequence_lengths)
        seq_len_string = "+".join(map(lambda i: str(i), self.__sequence_lengths))
        print(f"Constructed dataset of size {self.__total_len} ({seq_len_string})")
        if self.__total_len == 0:
            print("ERROR: empty dataset")
            from IPython import embed; embed()

    def __len__(self) -> int:
        """Number of rollouts/sequences in this dataset.
        """
        return self.__total_len

    def __get_sequence_of_length(
        self, index: int, sequence_length: int
    ) -> Tuple[torch.Tensor, ...]:
        """Returns a sequence of requested length.

        Args:
            index (int): Index of rollout to return.
            sequence_length (int): Length of rollout to return.

        Returns:
            Tuple[torch.Tensor, ...]: Tuple of tensors with length twice the number of features passed to constructor.
                For example if passed in features are "control", "state", it would return tuple of corresponding to:
                    1. control features
                    2. time offset of control features (with respect to smallest offset across features)
                    3. state features
                    4. time offset of state features (with respect to smallest offset across features)
        """
        sequence_idx = bisect_right(self.__sequence_start_idxs, index) - 1  # type: ignore
        rollout_idx = index - self.__sequence_start_idxs[sequence_idx]

        result: List[torch.Tensor] = []
        for f, delay in self.__features:
            result.append(self.processed_sequences[sequence_idx][f"{f}_{delay}"][rollout_idx:rollout_idx+sequence_length])
            result.append(self.processed_sequences[sequence_idx][f"time_{delay}"][rollout_idx:rollout_idx+sequence_length])

        return tuple(result)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Returns a rollout from a specific index.

        Args:
            index (int): Index of rollout to return.

        Returns:
            Tuple[torch.Tensor, ...]: Tuple of tensors with length twice the number of features passed to constructor.
                For example if passed in features are "control", "state", it would return tuple of corresponding to:
                    1. control features
                    2. time offset of control features (with respect to smallest offset across features)
                    3. state features
                    4. time offset of state features (with respect to smallest offset across features)
        """
        return self.__get_sequence_of_length(index, self.__sequence_length)

    @property
    def longest_rollout(self) -> Tuple[torch.Tensor, ...]:
        """

        Returns:
            Tuple[torch.Tensor, ...]: Tuple of tensors.
                Similar to __getitem__, except instead rollout being of length `sequence_length`, it's maximum possible length.
        """
        return self.__get_sequence_of_length(
            self.__sequence_start_idxs[self.__max_len_rollout_idx],
            self.__max_len_rollout,
        )

    @staticmethod
    def __parse_sequences(
        sequences: RawSequences,
        features: List[Tuple[str, int]],
        sequence_length: int,
        *args,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int], int, int]:
        """Parses synchronized sequences (i.e. there is a set of features for each timestamp).

        Args:
            sequences (RawSequences): Sequences to parse.
            features (List[Tuple[str, int]]): Features to extract from the sequence with the delay/offset for each feature.
            sequence_length (int): Length of each rollout (sequence in ML understanding) to extract.
                This is used to calculate number of possible rollouts.

        Returns:
            Tuple[List[Dict[str, torch.Tensor]], List[int], int, int]: Tuple of 4 elements containing:

                1. List of Dictionaries of processed sequences with proper delay synchronization.
                    Note that for saving data purposes they are still of full length and **not** sequence_length rollouts.
                    The indexing done it __get_sequence_of_length function retrieves rollouts from these sequences.
                2. List of number of rollouts for each sequence returned in the first element of this tuple.
                    Each sequence will contain different number of rollouts which are tracked here for ease of use.
                3. Index of sequence with the longest possible rollout.
                4. Length of the longest rollout across all sequences.
        """
        processed_sequences: List[Dict[str, torch.Tensor]] = []
        sequence_lengths: List[int] = []
        max_len_rollout = 0
        max_len_rollout_idx = 0

        max_delay = max(delay for _, delay in features)

        required_features = set(f for f, _ in features)

        seq_idx = 0

        for seq in sequences:
            if not required_features.issubset(seq.keys()):
                # Sequence does not contain all required features
                continue

            cur_seq = {}

            # Add torch sequences
            for f, delay in features:
                if delay == max_delay:
                    cur_seq[f"{f}_{delay}"] = torch.from_numpy(seq[f][delay:])
                    cur_seq[f"time_{delay}"] = torch.from_numpy(seq["time"][delay:] - seq["time"][:-delay])
                else:
                    cur_seq[f"{f}_{delay}"] = torch.from_numpy(seq[f][delay:-(max_delay - delay)])
                    cur_seq[f"time_{delay}"] = torch.from_numpy(seq["time"][delay:-(max_delay - delay)] - seq["time"][:-max_delay])

            # Check key is used for checking length of this sequence etc.
            # It should not be assumed to be a specific feature
            check_key = next(iter(cur_seq.keys()))

            # Calculate the number of sequences that can be read
            # In each step we are taking delay_steps steps forward (to get next element of the rollout)
            # for sequence_length steps
            # One rollout will be sequence_length long
            num_rollouts = (
                len(cur_seq[check_key]) - sequence_length + 1
            )

            # Sequence too short. No rollouts can be read
            if num_rollouts <= 0:
                continue

            # Add sequence to processed sequences
            processed_sequences.append(cur_seq)

            # Add it's length to the sequence lengths
            sequence_lengths.append(num_rollouts)

            # Determine whether it is the longest sequence
            if max_len_rollout < len(cur_seq[check_key]):
                max_len_rollout = len(cur_seq[check_key])
                max_len_rollout_idx = seq_idx

            # Only update seq_idx if we have a valid sequence
            seq_idx += 1

        return (
            processed_sequences,
            sequence_lengths,
            max_len_rollout_idx,
            max_len_rollout,
        )
