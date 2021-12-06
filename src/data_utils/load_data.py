import numpy as np
import h5py as h5
from pathlib import Path
from .hdf5_processing import hdf5_extract_data

def load_dataset(dataset_name: str):
    data_folder = Path("datasets") / dataset_name
    result = []

    # Numpy processing
    for numpy_path in data_folder.rglob("np.txt"):
        X = np.loadtxt(numpy_path)
        N = X.shape[0]
        seqs = []
        seq = []
        seq_no = 0
        for row in X:
            data = row[1:].reshape((1,-1))
            if row[0] != seq_no:
                if len(seq) > 1:
                    seqs.append(seq)
                seq = data
                seq_no = row[0]
            else:
                seq = np.concatenate([seq, data], 0)
        result.extend(seqs)

    # HDF5 Processing
    for file_path in list(data_folder.rglob("*.h5")) + list(data_folder.rglob("*.hdf5")):
        with h5.File(file_path) as f:
            result.extend(hdf5_extract_data(dataset_name, f))

    if not result:
        raise ValueError(
            f"Dataset: {dataset_name} not found. Ensure that is a folder under \'datasets/\' directory"
        )

    return result
