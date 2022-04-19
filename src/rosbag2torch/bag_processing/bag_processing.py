from pathlib import Path
from typing import Union

from .sequence_readers.abstract_sequence_reader import AbstractSequenceReader, Sequences


def load_bags(
    data_folder: Union[str, Path], reader: AbstractSequenceReader
) -> Sequences:
    for file_path in Path(data_folder).rglob("*.bag"):
        reader.extract_bag_data(file_path)

    sequences = reader.sequences
    reader.reset()

    return sequences
