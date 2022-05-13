from pathlib import Path
from typing import Union

from .sequence_readers.abstract_sequence_reader import AbstractSequenceReader, Sequences


def load_bags(
    data_folder: Union[str, Path], reader: AbstractSequenceReader
) -> Sequences:
    """Function to load the data from a folder of bag files using a provided reader.

    Args:
        data_folder (Union[str, Path]): Folder to extract data from.
        reader (AbstractSequenceReader): Reader to extract bags with.

    Returns:
        Sequences: A list of sequences (each is a dictionary) extracted from the bag files.
            Note that one bag may contain any number of sequences (0, 1, or more).
    """
    for file_path in Path(data_folder).rglob("*.bag"):
        reader.extract_bag_data(file_path)

    sequences = reader.sequences
    reader.reset()

    return sequences
