import numpy as np
import h5py as h5
from pathlib import Path


def h5_extract_data(dataset_name: str, h5_file: h5.File):
    if "rzr" in dataset_name:
        # 1. Extract data
        # D
        topics = [
            # D
            "input_pair/control/steer", "input_pair/control/throttle", "input_pair/control/steer",
            # H
            "input_pair/state/twist/linear", "input_pair/state/twist/angular",
            # P
            "input_pair/state/pose/position", "input_pair/state/pose/orientation", "input_pair/state/twist/linear", "input_pair/state/twist/angular",
        ]
        sim_time = h5_file["input_pair/timestamp"][:].squeeze()
        sim_data_raw = []
        for topic in topics:
            topic_data = h5_file[topic][:]
            sim_data_raw.append(topic_data)
        sim_data_raw = np.concatenate(sim_data_raw, axis=1)

        # H
        # linear.x
        sim_data_raw[:, len(topics) + 1] = h5_file["input_pair/state/twist/linear"][:, 0]
        # angular.z
        sim_data_raw[:, len(topics) + 1] = h5_file["input_pair/state/twist/angular"][:, 2]

        # P

        # Planer

        print("State")

        # 2. Filter over large timestamp differences to create sequences
        diff = sim_time[1:] - sim_time[:-1]

        # TODO: Make an optional arg in main argparse?
        action_frequency = 10
        max_allowed_diff = (1.0 / action_frequency) * 1.2  # 20% Seems like a loose enough constraint
        seq_breaks = np.argwhere(diff > max_allowed_diff).squeeze()

        sim_seqs = []
        seq_start = 0
        print(seq_breaks)
        for seq_break in seq_breaks:
            seq_end = seq_break + 1
            sim_seqs.append(sim_data_raw[seq_start:seq_end])
            seq_start = seq_break + 1
        # Last sequence
        sim_seqs.append(sim_data_raw[seq_start:])

        # 3. Same for synced vehicle input state?
        # TODO: This will be a superset of input_pair.
        # It might be better to just log this stuff.

        return sim_seqs
    else:
        raise ValueError(f"Dataset {dataset_name} does not have a corresponding extraction script.\n"
        "Please implement it in load_data.py h5_extract_data function")

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
                seqs.extend(h5_extract_data(dataset_name, f))
        return seqs

    except Exception as e:
        raise ValueError(
            f"Dataset: {dataset_name} not found. Ensure that is a folder under \'datasets/\' directory"
        )