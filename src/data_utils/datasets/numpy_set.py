from typing import Dict, List

import numpy as np
from .named_dataset import NamedDataset

class NumpyDataset(NamedDataset):
    name = "numpy"

    def __init__(
        self,
        seqs: List[Dict[str, np.ndarray]],
        x_features: List[str],
        y_features: List[str],
    ) -> None:
        # Convert sequences from List[Dict[str, np.ndarray]] to List[[np.ndarray]]
        # With ordering from model
        self.numpy_seqs = []

        # Keep track of size of x_features (D) and y_features (H), as they are used by 
        self.D = None
        self.H = None

        for s in seqs:
            self.numpy_seqs.append(np.concatenate([s[f] for f in x_features + y_features], axis=1))

            s_D = sum([s[f].shape[1] for f in x_features])
            s_H = sum([s[f].shape[1] for f in y_features])

            if self.D is None:
                self.D = s_D
            else:
                assert self.D == s_D, "Shape D (x_features) doesn't match between sequences"
            if self.H is None:
                self.H = s_H
            else:
                assert self.H == s_H, "Shape H (y_features) doesn't match between sequences"

        # Define variables needed for numpy models

        # That's it. Transformations of arrays are done in `train` function of LinearModel

    @property
    def dataset(self) -> List[np.ndarray]:
        return self.numpy_seqs
