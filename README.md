# dynamics-modeling
This repo deals with machine learning (ML) for dynamics models for the RZR (and Warthog? (TODO: Verify it works for Warthog bags too)) vehicles in simulation and on the real vehicles.

## Installation

```
git clone https://github.com/balbok0/dynamics-modeling.git
cd dynamics-modeling
pip install .
```

## Usage
Example usage files are in the `examples` folder.

Below is a list of features that rosbag2torch provides

### Reading bag files into sequences
```python
from rosbag2torch import load_bags, readers, filters, transforms

# Reader is a class that will subscribe to appropriate topics and read them into a sequence.
reader = readers.FixedIntervalReader(
    # features to read. Each feature has a corresponding transform which will specify what topics it requires.
    # This lookup is done automatically by reader.
    ["state", "control", "target"],
    # Filters are used to filter out regions of the bag that are not of interest.
    # In order for a given timestamp to be considered, it must pass all filters.
    filters=[
        # Forward filter ensures that vehicle is driving forward.
        filters.ForwardFilter(),
        # PIDInfo Filter ensures that vehicle is in autonomous mode, with brake working and sequence controller healthy.
        filters.PIDInfoFilter(),
    ],
    # Field specific to FixedIntervalReader.
    # It will fit a spline through data, and record states at the specified interval.
    log_interval=1.0 / 30.,
)

# Load bags takes a folder of bag files, and a reader.
# It will apply the reader for each bag file in the folder.
sequences = load_bags("path_to_folder_with_bags", reader)

# NOTE: You can reuse the reader for multiple load_bags calls.
# This can be useful if you have separate folders for train and validation sets.
```

### Converting sequences into Dataset
```python
from rosbag2torch import SequenceLookaheadDataset

# This assumes that code from previous section was run, and sequences is a variable.

# SequenceLookaheadDataset is a Dataset that will take features at index t and look ahead for delayed features at time t + delay_steps.
# Then it will do the same for indexes t + delay_steps and t + 2 * delay_steps, creating a sequence at timestamps:
# t, t + delay_steps, t + 2 * delay_steps, ... of length sequence_length.
dataset = SequenceLookaheadDataset(
    # Result of load_bags call
    sequences,
    # Features to get at time t.
    features=["state", "control"],
    # Features to get at time t + delay_steps
    delayed_features=["target"],
    # Number of steps to look ahead.
    delay_steps=3,
    # Size of each sequence
    sequence_length=100,
)

# Voila, now you can iterate over the dataset PyTorch style.
from torch.utils.data import DataLoader

# Dataset will yield a tuple that contains:
#   - all of features (in order given to constructor)
#   - all of delayed features (in order given to constructor)
#   - The difference in time at which the delayed features and the features were logged.
for state, control, target, dts in Dataloader(dataset):
    pass
```

## Development
From this point on-ward are instructions and other resources for the development.

### Setup Dev Environment
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) (or some other conda version)
2. Clone this repo and `cd` into folder it's cloned into.
3. Run `conda env create -f enviornment.yaml`

This will create environment named `dynamics-modeling`. All of the following instructions assume that you are in that environment (you can activate it by `conda activate dynamics-modeling`).


### Structure
**NOTE This section is out of date**

**NOTE:** This repo is still Work in Progress, so the structure might change a lot. We will try to keep README updated, but it might be outdated.

Repo is split into datasets and code.
Datasets should be put in *datasets* folder and will be automatically read from it.

Code is split into few sub-folders based on stages:
- [Data Loading](src/data_utils/README.md)
- [Models](src/models/README.md)
- [Optimization Logic/Loop](src/optimization_logic/README.md)

Lastly the scripts `src/main.py` combines all of these building blocks into one script that loads data/models, trains them and saves the model to a specified destination. For expected arguments either run `python src/main.py --help` or see [*src/parse_args.py*](src/parse_args.py).


## TODOs
- [ ] Fix other datasets to follow SequenceLookaheadDataset __get_item__ structure.
- [ ] Invoke tasks for linting and testing(? See below)
- [ ] Move to GitLab
- [ ] Dask support for arrays bigger than memory? I don't think it will be an issue with dynamics-modeling (unless we do some stuff with elevation map), but it might be useful for other extractions.

### Old TODOs
- [x] Make validation/plotting pipeline.
- [x] Debug of current differential pose approach, and figuring out sync issues.
- [x] Verify that robot type argument is even necessary. If it is then make sure it's properly passed into data loading, models etc.
- [x] Add support for AutoRally style input/output.
- [x] DOCUMENTATION. More an ongoing task
- [x] Configuration files.
- [x] Split code into library and script. (For example datasets/parsing data should be in library, but loading args/models would in script)
