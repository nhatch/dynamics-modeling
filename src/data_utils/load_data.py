from functools import lru_cache
from typing import Dict, List, Type
import numpy as np
import h5py as h5
from pathlib import Path
from inspect import isclass, ismodule

from data_utils.datasets.named_dataset import NamedDataset

from .bag_processing.bag_processing import load_bag
from .hdf5_processing import hdf5_extract_data

SequenceData = Dict[str, np.ndarray]  # TODO: For really big datasets, this should also have a possible type of Callable[[], SequenceData]
# TODO: Also long reach, but we should be requiring "time" to be a key. This way

def load_dataset(dataset_name: str, x_features: List[str], y_features: List[str], robot_type: str, dataset_type: str) -> List[SequenceData]:
    data_folder: Path = Path("datasets") / dataset_name

    result = []

    # Numpy processing
    result.extend(load_numpy(data_folder, dataset_name, x_features + y_features, robot_type))

    # HDF5 Processing
    # TODO: Detect if hdf5 file is ackermann or skid. Add parsing for skid
    if robot_type == "ackermann":
        for file_path in list(data_folder.rglob("*.h5")) + list(data_folder.rglob("*.hdf5")):
            with h5.File(file_path) as f:
                result.extend(hdf5_extract_data(dataset_name, f))

    # Bag Processing
    result.extend(load_bag(data_folder, dataset_name, x_features + y_features, robot_type))

    if not result:
        raise ValueError(
            f"Dataset: {dataset_name} not found. Ensure that is a folder under \'datasets/\' directory"
        )

    return convert_data_to_dataset(result, dataset_type, x_features, y_features)


def load_numpy(data_folder: Path, dataset_name: str, features: List[str], robot_type: str):
    seqs = []

    # This is legacy stuff, so we can specifically do this for FIXED datasets
    if dataset_name in {"rzr_sim", "sim_data", "sim_odom_twist"}:
        # Ensure that features requested is a subset of control, state, target
        numpy_features = {"control", "state", "target"}
        if not numpy_features.issuperset(features):
            # Features is not a subset of ones that numpy can provide
            print(features)
            return []

        if dataset_name == "sim_data":
            D = 2 # cmd_vel in the form dx, dtheta
            H = 0
            # P = 3 for poses (x, y, theta)
            P = 3
        elif dataset_name == "sim_odom_twist":
            D = 2
            # This also includes odom measurements of dx, dtheta
            H = 2
            # This also includes dx, dy, dtheta
            P = 6
        elif dataset_name == "rzr_sim":
            D = 3 # throttle, brake, steer (multiplied by -1 if we're in reverse)
            H = 2
            P = 6

        for numpy_path in data_folder.rglob("np.txt"):
            X = np.loadtxt(numpy_path)
            N = X.shape[0]
            seq = []
            seq_no = 0
            for row in X:
                data = row[1:].reshape((1,-1))
                if row[0] != seq_no:
                    if len(seq) > 1:
                        # Extract all of the features
                        tmp = {
                                "control": seq[:, :D],
                                "state": seq[:, D:D + H],
                                "target": seq[:, -P:],
                        }
                        # Put them in an ordered pashion
                        seqs.append(
                            {
                                "time": np.ones(len(seq)),
                                **{f: tmp[f] for f in features}
                            }
                        )
                    seq = data
                    seq_no = row[0]
                else:
                    seq = np.concatenate([seq, data], 0)
    return seqs


def get_all_datasets(module = None, explored_modules = None) -> Dict[str, Type[NamedDataset]]:
    def filter_for_models_with_spec(k, v):
        return (
            not k.startswith("__")  # __main__ etc.
            and isclass(v)
            and not k == "NamedDataset"
            and issubclass(v, NamedDataset)
        )

    if module is None:
        from . import datasets
        module = datasets
    if explored_modules is None:
        explored_modules = set()

    explored_modules.add(module)

    result = dict()
    for k, v in vars(module).copy().items():
        if ismodule(v) and v not in explored_modules:
            recurse = get_all_datasets(v, explored_modules)
            result |= recurse
        elif filter_for_models_with_spec(k, v):
            result[v.name] = v
    return result

def convert_data_to_dataset(data: List[SequenceData], dataset_type: str, x_features: List[str], y_features: List[str]):
    datasets_types = get_all_datasets()

    if dataset_type not in datasets_types:
        raise ValueError(
            f"Model named {dataset_type} not found.\n"
            f"Available models are {', '.join(datasets_types.keys())}"
        )
    return datasets_types[dataset_type](
        seqs=data,
        x_features=x_features,
        y_features=y_features,
    )
