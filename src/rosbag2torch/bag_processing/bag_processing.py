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
    p = Path(data_folder)
    if not p.is_dir():
        raise NotADirectoryError("{} is not a directory".format(data_folder))

    for file_path in p.rglob("*.bag"):
        reader.extract_bag_data(file_path)

    sequences = reader.sequences
    reader.reset()

    if len(sequences) == 0:
        print("Warning: reader found no sequences in {}".format(data_folder))
    else:
        print("Found {} sequences of len {}".format(
            len(sequences), list(map(lambda s: len(s["time"]), sequences))))

    return sequences
