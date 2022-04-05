from typing import List
from pathlib import Path
from .sequence_readers.abstract_sequence_reader import AbstractSequenceReader
from .sequence_readers import ASyncSequenceReader, FixedIntervalReader
from .filters import ForwardFilter, PIDInfoFilter


def load_bag(data_folder: Path, dataset_name: str, features: List[str], robot_type: str):
    # filters = [ForwardFilter(), PIDInfoFilter()]
    filters = [ForwardFilter(), PIDInfoFilter()]

    # TODO: Add parsing for skid. Bags should also be able to just subscribe to cmd_vel
    reader: AbstractSequenceReader
    reader = ASyncSequenceReader(features, features_to_record_on=["control"], filters=filters)
    # reader = FixedIntervalReader(features, filters=filters)

    if robot_type == "ackermann":
        for file_path in data_folder.rglob("*.bag"):
            reader.extract_bag_data(file_path)

    return reader.sequences
