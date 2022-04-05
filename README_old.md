
This repo deals with machine learning (ML) for dynamics models for the Warthog and RZR vehicles,
both in simulation and on the real vehicles. This involves:
- rosbag-recording of the proper topics
- for real-world data: obtaining ground truth state (this may be computed later rather than recorded in real time)
- extraction of data from rosbags and ground truth into an ML-friendly format
- definitions for various potential dynamics models (kinematic skid steer, Ackermann, neural networks)
- scripts for training the parameters of these models
- exporting of the trained parameters into a format that can be used within the Phoenix stack

Not all of these functions have been implemented yet. So far it has only been used to fit the
"effective wheel base" parameter of the skid-steer Warthog kinematic model. See TODO.txt for a
list of things that have been done / planned to do / abandoned.


## To collect an ML-friendly simulator dataset:

1. Start the simulator.
2. Run `dyndata.sh` (from anywhere you can record ROS topics; if you're using Docker this may need to be inside the Docker container). This will begin recording a rosbag.
3. Use Rviz to execute however many trajectories you want. (NOTE: Do not use manual control. It seems like Unity does not publish manual control commands to ROS, so we can't record the control input.)
4. Stop the dyndata.sh script (ctrl-C).
5. From this directory, run `extract_wh.sh BAGFILE DATASET_NAME`. (Use `extract_rzr.sh` instead for the RZR). This extracts the relevant data into CSV format. This can take a while because it loops through the whole bagfile for each topic.
6. From this directory, run `./extract_ml.py DATASET_NAME`. This converts the many CSV files into a single `np.txt` 2D array with one row per "training sample".

The number of columns of this array depends on which type of robot collected the data. In general, it will have the following columns:

- one "sequence number". The bag files are split into "sequences" based on the timestamps of the recorded control inputs. (When the Phoenix control stack is not running, usually nothing is published on the control topics. So we can use this to segment trajectories.) The sequence number increments by one for each trajectory.
- the control input (2D for Warthog: dx, dtheta; 3D for RZR: throttle, brake, steering)
- odom measurement for dx, dtheta
- ground truth pose (x, y, theta)
- ground truth twist (dx, dy, dtheta)

## To collect a real-world dataset:

I'm not sure; I haven't done this yet. In particular, obtaining ground truth data will be difficult.

## To train a dynamics model

Run `./train.py DATASET_NAME`.

Currently this only works for Warthog datasets. It will fit various models (currently all kinematic; the most complicated one just fits a linear model from control dx,dtheta to actual dx,dy,dtheta), and print their validation loss and fitted parameters to stdout.


## TODO:

- [] Plotting that takes in sequence data
- []