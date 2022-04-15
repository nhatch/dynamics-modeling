import numpy as np
import scipy.spatial.transform as trf


def planar_pose(pose: np.ndarray) -> np.ndarray:
    if len(pose.shape) == 1:
        process_pose = pose.reshape((1, len(pose)))
    else:
        process_pose = pose

    r1 = trf.Rotation.from_quat(process_pose)
    rot = r1.as_euler('zyx')

    result = np.array([process_pose[:, 0], process_pose[:, 1], rot[:, 0]])
    # Flatten
    if len(pose.shape) == 1:
        result = result[:, 0]

    return result

def planar_twist(twist: np.ndarray, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    if len(twist.shape) == 1:
        process_twist = twist.reshape((1, len(twist)))
    else:
        process_twist = twist

    # We need the `pose` because `twist` is given in the world frame,
    # while we want it in the robot base frame.
    linear_vel = np.cos(theta) * process_twist[:, 0] + np.sin(theta) * process_twist[:, 1]
    # Transverse vel should be zero except for projection errors and
    # times when the robot is slipping sideways.
    transverse_vel = -np.sin(theta) * process_twist[:, 0] + np.cos(theta) * process_twist[:, 1]

    result = np.array([linear_vel, transverse_vel, theta_dot])
    # Flatten
    if len(twist.shape) == 1:
        result = result[:, 0]

    return result