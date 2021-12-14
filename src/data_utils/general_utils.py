import numpy as np
import scipy.spatial.transform as trf


def planar_pose(pose: np.ndarray) -> np.ndarray:
    r1 = trf.Rotation.from_quat(pose)
    rot = r1.as_euler('zyx')
    return np.array([pose[:, 0], pose[:, 1], rot[:, 0]])

def planar_twist(twist: np.ndarray, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
    # We need the `pose` because `twist` is given in the world frame,
    # while we want it in the robot base frame.
    linear_vel = np.cos(theta) * twist[:, 0] + np.sin(theta) * twist[:, 1]
    # Transverse vel should be zero except for projection errors and
    # times when the robot is slipping sideways.
    transverse_vel = -np.sin(theta) * twist[:, 0] + np.cos(theta) * twist[:, 1]
    return np.array([linear_vel, transverse_vel, theta_dot])