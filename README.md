# dynamics-modeling
This repo deals with machine learning (ML) for dynamics models for the RZR (and Warthog? (TODO: Verify it works for Warthog bags too)) vehicles,
bot in simulation and on the real vehicles.

## Setup
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) (or some other conda version)
2. Clone this repo and `cd` into folder it's cloned into.
3. Run `conda env create -f enviornment.yaml`

This will create environment named `dynamics-modeling`. All of the following instructions assume that you are in that environment (you can activate it by `conda activate dynamics-modeling`).

## Usage
To train a model on specific dataset:
1. Create a new folder under *datasets* folder. For purposes of this example we will name it *new_dataset*. Put all bags of interest in that folder.
2. Make sure that bag contains robot type currently supported. (right now ackermann or skid) (TODO: Do we need this? I believe right now we just ignore this argument. It's for legacy purposes.) To look for supported robots see `python src/main.py --help`
3. Choose a model from output of `python src/main.py --list-models`. For purposes of this example we will use [*linear*](src/models/torch_models/linear.py)
4. Run `python src/main.py` with proper arguments. For example: `python src/main.py -m linear -d new_dataset -r ackermann`.


## Structure
**NOTE:** This repo is still Work in Progress, so the structure might change a lot. We will try to keep README updated, but it might be outdated.

Repo is split into datasets and code.
Datasets should be put in *datasets* folder and will be automatically read from it.

Code is split into few sub-folders based on stages:
- [Data Loading](src/data_utils/README.md)
- [Models](src/models/README.md)
- [Optimization Logic/Loop](src/optimization_logic/README.md)

Lastly the scripts `src/main.py` combines all of these building blocks into one script that loads data/models, trains them and saves the model to a specified destination. For expected arguments either run `python src/main.py --help` or see [*src/parse_args.py*](src/parse_args.py).


## TODOs
- [ ] Make validation/plotting pipeline. This requires a debug of current differential pose approach, and figuring out sync issues.
- [ ] Verify that robot type argument is even necessary. If it is then make sure it's properly passed into data loading, models etc.
- [ ] Add support for AutoRally style input/output.
- [ ] DOCUMENTATION. More an ongoing task
- [ ] Configuration files.
- [ ] Invoke tasks for linting and testing(? See below)
- [ ] Figure out some integration tests. Maybe a simple train loop
- [ ] Move to GitLab
- [ ] Split code into library and script. (For example models/datasets/parsing data should be in library, but loading args would in script)
- [ ] Dask support for arrays bigger than memory? I don't think it will be an issue with dynamics-modeling (unless we do some stuff with elevation map), but it might be useful for other extractions.