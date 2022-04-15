import rosbag2torch
from .numpy_set import NumpyDataset

datasets = {
    "torch_lookahead": rosbag2torch.LookaheadSequenceDataset,
    "torch_lookahead_diff": rosbag2torch.LookaheadDiffSequenceDataset,
    "numpy": NumpyDataset,
}

filters = {
    "forward": rosbag2torch.filters.ForwardFilter,
    "pid_info": rosbag2torch.filters.PIDInfoFilter,
}

readers = {
    "async": rosbag2torch.readers.ASyncSequenceReader,
    "fixed_interval": rosbag2torch.readers.FixedIntervalReader,
}
