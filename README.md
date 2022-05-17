# dynamics-modeling
This repo deals with machine learning (ML) for dynamics models for the RZR (and Warthog? (TODO: Verify it works for Warthog bags too)) vehicles in simulation and on the real vehicles.

It is split into two parts:

1. rosbag2torch - a general purpose library that converts rosbags to torch datasets.
2. examples/dynamics_modeling - a collection of examples that use the library to train dynamics modeling models.

## Installation

```
git clone https://github.com/balbok0/dynamics-modeling.git
cd dynamics-modeling
pip install .
```

## rosbag2torch


### Usage
More usage examples files are in the `examples` folder.

Below is a list of features that rosbag2torch provides

#### Reading bag files into sequences
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

#### Converting sequences into Dataset
```python
from rosbag2torch import SequenceLookaheadDataset

# This assumes that code from previous section was run, and sequences is a variable.

# SequenceLookaheadDataset is a Dataset that will take features at index t + offset (specified for each feature separately).
# Then it will form sequences of length sequence_length.
# For the example below (if c is control, s is state, and t is target), it will form sequences:
# [(c0, s3, t4), (c1, s4, t5), (c2, s4, t5), ...]
# [(c1, s4, t5), (c2, s4, t5), (c3, s5, t6), ...]
# NOTE: In the dataloader in few lines you will see that also time offsets for each feature are included.
dataset = SequenceLookaheadDataset(
    # Result of load_bags call
    sequences,
    # Features to get at time t + offset (specified as a second argument to the tuple).
    features=[
        ("control", 0),
        ("state", 3),
        ("target", 4),
    ]
    # Size of each sequence
    sequence_length=100,
)

# Voila, now you can iterate over the dataset PyTorch style.
from torch.utils.data import DataLoader

# Dataset will yield a tuple that contains:
#   - all of features (in order given to constructor)
#   - the difference in time of the feature delay/offset and the "0" offset (the feature with the least delay).
# NOTE: As a result of the above, at least one of the elements in the tuple will be always 0, which makes it useless.
#       This approach however gives freedom when there are non-trivial delay relations between different sets of features.
for control, _, state, state_delay, target, target_delay in DataLoader(dataset):
    # In this case control_delay is always 0, so we just ignore it.
    pass
```

## Examples/Dynamics Modeling
As of time of writing there are 4 runnable files in the *examples* folder.
They all train a model that is given a control (throttle, brake, steer) and state (dx, dtheta) and predicts the acceleration (ddx, ddtheta).
Target values (y's in ML) are given as (dx, dtheta) after some time dt (also provided).
However, they all use different approaches to training these models:

1. *hidden2relu* - For each time step, the model takes the state and control and predicts the acceleration.
   It then applies this acceleration to get the state after some dt. Loss is defined between this predicted state and the target state.
2. *sequence_model* - Similar to *hidden2relu*, but applies the model to a sequence of controls, and a start state.
   It then predicts the acceleration, which is used to predict the state after some dt. That state is fed back into a model, creating a rollout of states/controls.
   Loss is defined for at each point of the rollout with corresponding target state.
3. *hyperparameter_search_sequence_model* - Similar to *sequence_model*, but uses a hyperparameter search to find the best model.
4. *xy_hyperparameter_search_sequence_model* - Same as *hyperparameter_search_sequence_model*, but with different loss definition.
   Rather than defining loss in space (dx, dtheta) it rolls out these values in (x, y, theta) space and defines loss on top of these.

Both of the hyperparameter search scripts use [tensorboard](https://www.tensorflow.org/tensorboard/) to log the results.

There are also additional scripts:

* example_utils - a collection of functions that are used in the examples. This is mainly used because there is a lot of code overlap between all sequence scripts.
* look_at_dists - A script that can be used to look at the distribution of the training data.
* unit_tests - Script that can perform some controlled tests given a scripted model. TODO: These should not be here I think. Ideally we should just have tests folder for dynamics modeling.

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
- [ ] Utility to loop over .h5cache files along with bags.
- [ ] Test & Invoke tasks for testing
- [ ] Decouple looking up transforms when extracting features in reader.
- [ ] Move to GitLab
- [ ] Dask support for arrays bigger than memory? I don't think it will be an issue with dynamics-modeling (unless we do some stuff with elevation map), but it might be useful for other extractions.

### Old TODOs
- [x] Register hook for transforms
- [x] Make validation/plotting pipeline.
- [x] Fix other datasets to follow SequenceLookaheadDataset __get_item__ structure.
- [x] Debug of current differential pose approach, and figuring out sync issues.
- [x] Verify that robot type argument is even necessary. If it is then make sure it's properly passed into data loading, models etc.
- [x] Add support for AutoRally style input/output.
- [x] DOCUMENTATION. More an ongoing task
- [x] Configuration files.
- [x] Split code into library and script. (For example datasets/parsing data should be in library, but loading args/models would in script)
