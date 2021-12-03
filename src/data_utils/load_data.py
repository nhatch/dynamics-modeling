import numpy as np
import h5py as h5
from pathlib import Path
from .hdf5_processing import hdf5_extract_data

def load_dataset(dataset_name: str):
    try:
        X = np.loadtxt('datasets/' + dataset_name + '/np.txt')
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
        return seqs
    except OSError as e:
        # dataset not found
        # Try h5py
        data_folder = Path("datasets") / dataset_name
        # Glob only supports wildcards not regex
        file_paths = list(data_folder.glob("*.h5")) + list(data_folder.glob("*.hdf5"))

        seqs = []

        for file_path in file_paths:
            with h5.File(file_path) as f:
                seqs.extend(hdf5_extract_data(dataset_name, f))
        return seqs

    except Exception as e:
        raise ValueError(
            f"Dataset: {dataset_name} not found. Ensure that is a folder under \'datasets/\' directory"
        )