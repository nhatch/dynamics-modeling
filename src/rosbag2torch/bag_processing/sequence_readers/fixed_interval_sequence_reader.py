from pathlib import Path
from typing import List, Union

import numpy as np
from scipy import interpolate

from ..filters import AbstractFilter
from .abstract_sequence_reader import AbstractSequenceReader, Sequence, Sequences


class FixedIntervalReader(AbstractSequenceReader):
    def __init__(
        self,
        required_keys: List[str],
        log_interval: float = 0.1,
        spline_pwr: int = 3,
        filters: List[AbstractFilter] = [],
        *args,
        **kwargs
    ) -> None:
        super().__init__(required_keys, filters, *args, **kwargs)

        self.log_interval = log_interval
        self._sequences: Sequences = []
        self.spline_pwr = spline_pwr

    @property
    def sequences(self) -> Sequences:
        return self._sequences

    def _transform_raw_sequences(self):
        for raw_sequence in self.cur_bag_raw_sequences:
            min_ts, max_ts = None, None
            is_empty_sequence = False
            for feature_name in self.required_keys:
                # Sequence is empty
                if len(raw_sequence[feature_name][0]) <= 2:
                    is_empty_sequence = True
                    break

                if min_ts is None or raw_sequence[feature_name][1][0] < min_ts:
                    min_ts = raw_sequence[feature_name][1][0]
                if max_ts is None or raw_sequence[feature_name][1][-1] > max_ts:
                    max_ts = raw_sequence[feature_name][1][-1]

            # Empty sequence break
            if is_empty_sequence:
                break

            # Goal timestamps (that sequence is actually logged on)
            # Add half of log interval to also include the last message
            goal_ts = (
                np.arange(min_ts, max_ts + self.log_interval / 2, self.log_interval)
                - min_ts
            )

            cur_sequence: Sequence = {}

            # For each feature interpolate each column to have cadence of the log interval
            for feature_name in self.required_keys:
                # Get the timestamps and values. Make timestamps relative to the start of the sequence
                feature_data, feature_ts = raw_sequence[feature_name]
                feature_ts -= min_ts

                # NOTE: What are knots used for?
                knots = np.linspace(
                    feature_ts[0], feature_ts[-1], int(round(feature_ts.size / 10.0))
                )[1:-1]

                # Pre-allocate array
                processed_feature_sequence = np.zeros(
                    (len(goal_ts), feature_data.shape[1])
                )

                # For each column get spline params and interpolate back onto the fixed cadence goal timestamps
                for col_idx in range(feature_data.shape[1]):
                    spline_params = interpolate.splrep(
                        feature_ts, feature_data[:, col_idx], k=self.spline_pwr, t=knots
                    )
                    processed_feature_sequence[:, col_idx] = interpolate.splev(
                        goal_ts, spline_params
                    )

                cur_sequence[feature_name] = processed_feature_sequence

            # Lastly add timestamp of the sequence
            cur_sequence["time"] = goal_ts + min_ts

            self._sequences.append(cur_sequence)

    def extract_bag_data(self, bag_file_path: Union[str, Path]):
        self._extract_raw_sequences(bag_file_path)
        self._transform_raw_sequences()
        self.end_bag()

    def reset(self):
        self._sequences = []
