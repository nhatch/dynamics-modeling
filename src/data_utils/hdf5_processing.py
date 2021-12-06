import h5py as h5
import numpy as np
import scipy.spatial.transform as trf


def gear_to_multiplier(gearbox_gear, gearbox_mode):
    # Very roundabout, but fast way to map mode to -1, 0, 1
    mode_to_direction_dict = {
        0: 1.0,  # Manual
        1: 0.0,  # Park
        2: -1.0,  # Rear
        3: 0.0,  # Neutral
        4: 1.0,  # Drive
        5: 1.0,  # Low Drive
    }
    def mode_to_dir_iter(a):
        return mode_to_direction_dict[int(a)]
    mode_to_dir = np.vectorize(mode_to_dir_iter)

    # FIXME: They should match on majority of experiments, but lead to very different performances.

    # return mode_to_dir(gearbox_mode)
    return gearbox_gear


def hdf5_extract_data(dataset_name: str, h5_file: h5.File):
    if "rzr" in dataset_name:
        # 1. Extract data
        topics_ground_truth = ["input_pair/state/pose/position", "input_pair/state/pose/orientation", "input_pair/state/twist/linear", "input_pair/state/twist/angular",]

        sim_time = h5_file["input_pair/timestamp"][:].squeeze()
        sim_data_raw = []

        # D (control)
        topics_control = ["input_pair/control/steer", "input_pair/control/throttle", "input_pair/control/brake",]
        for topic in topics_control:
            sim_data_raw.append(h5_file[topic][:])


        # Map steer, throttle to proper direction
        direction = gear_to_multiplier(
            h5_file["synced_vehicle_input_state/state/gearbox_gear"][:],
            h5_file["synced_vehicle_input_state/state/gearbox_mode"][:]
        )
        sim_data_raw[1] *= direction
        sim_data_raw[0] *= direction

        # TODO: Should we do anything "breaking phase" here?

        # H (state)
        sim_data_raw.append(h5_file["input_pair/state/twist/linear"][:, 0])  # linear.x
        sim_data_raw.append(h5_file["input_pair/state/twist/angular"][:, 2])  # angular.z

        # P (ground truth)
        # FIXME: We should actually record ground truth in the simulation in HDF5

        def planar_pose(pose: np.ndarray) -> np.ndarray:
            r1 = trf.Rotation.from_quat(pose)
            rot = r1.as_euler('zyx')
            return np.array([pose[:, 0], pose[:, 1], rot[:, 0]])

        def planar_twist(twist, theta, theta_dot):
            # We need the `pose` because `twist` is given in the world frame,
            # while we want it in the robot base frame.
            linear_vel = np.cos(theta) * twist[:, 0] + np.sin(theta) * twist[:, 1]
            # Transverse vel should be zero except for projection errors and
            # times when the robot is slipping sideways.
            transverse_vel = -np.sin(theta) * twist[:, 0] + np.cos(theta) * twist[:, 1]
            return np.array([linear_vel, transverse_vel, theta_dot])

        p_pose = planar_pose(h5_file["input_pair/state/pose/orientation"][:]).T
        p_twist = planar_twist(h5_file["input_pair/state/twist/linear"][:], p_pose[:, 2], h5_file["input_pair/state/twist/angular"][:, 2])

        sim_data_raw.extend(p_pose.T)
        sim_data_raw.extend(p_twist)

        # Assert all columns of the same size
        assert [len(sim_data_raw[0]) == len(sim_data_raw[i]) for i in range(1, len(sim_data_raw))]

        tmp = np.zeros((len(sim_data_raw[0]), len(sim_data_raw)))
        for i, r in enumerate(sim_data_raw):
            tmp[:, i] = r.squeeze()
        sim_data_raw = tmp

        # 2. Filter over large timestamp differences to create sequences
        diff = sim_time[1:] - sim_time[:-1]

        # TODO: Make an optional arg in main argparse?
        action_frequency = 10
        max_allowed_diff = (1.0 / action_frequency) * 1.2  # 20% Seems like a loose enough constraint
        seq_breaks = np.argwhere(diff > max_allowed_diff).squeeze()

        sim_seqs = []
        seq_start = 0
        print(seq_breaks)
        for seq_break in seq_breaks:
            seq_end = seq_break + 1
            sim_seqs.append(sim_data_raw[seq_start:seq_end])
            seq_start = seq_break + 1
        # Last sequence
        sim_seqs.append(sim_data_raw[seq_start:])

        # 3. Same for synced vehicle input state?
        # TODO: This will be a superset of input_pair.
        # It might be better to just log this stuff.

        return sim_seqs
    else:
        raise ValueError(f"Dataset {dataset_name} does not have a corresponding extraction script.\n"
        "Please implement it in load_data.py h5_extract_data function")
