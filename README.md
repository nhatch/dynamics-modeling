To collect a simulator dataset:

1. Start the simulator.
2. Run `dyndata.sh` (from inside the docker container, i.e. anywhere you can record ROS topics).
3. Use Rviz or manual control to execute however many trajectories you want.
4. Stop the dyndata.sh script.
5. Run `extract.sh BAGFILE DATASET_NAME` (from inside the docker container)
6. From outside the docker container, move the folder DATASET_NAME into this repo.

To run the stuff in this repo [TODO]:

1. `extract_ml.sh`
2. `train.sh`
