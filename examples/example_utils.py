import numpy as np
from typing import Optional


def reconstruct_poses_from_odoms(d_odom: np.ndarray, dt: np.ndarray, start_pose: Optional[np.ndarray] = None):
    """This function reconstructs the trajectory from odometry data.
    Odometry data is assumed to be in the form of [v_x, v_y, v_theta] or [v_x, v_theta],
    with dx and dy being in the robot frame.

    Args:
        d_odom (np.ndarray): A (n, 2) or (n, 3) array of odometry data.
        dt (np.ndarray): A (n,) array of differences in timestamps.
            It needs take delay_steps into account.
            This means that: dt[i] = t[i + delay_steps] - t[i]
        start_pose (Optional[np.ndarray], optional): A (3,) array of the starting pose (x, y, theta).
            Defaults to (0, 0, 0).

    Returns:
        np.ndarray: An (n + 1, 3) array of poses.
            Row i + 1 is the pose after d_odom[i] has been applied.
            The first row is the start pose.
            Columns correspond to (x, y, theta)
    """
    # Asserts to double check that everything is in the right format
    assert len(d_odom.shape) == 2 and d_odom.shape[1] in {2, 3}, \
        f"d_odom must be a 2D array with 2 (dx, dtheta) or 3 (dx, dy, dtheta) columns. Instead it is of shape {d_odom.shape}"
    # Default value for start_pose + assert if specified
    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])
    else:
        assert len(start_pose.shape) == 1 and start_pose.shape[0] == 3, \
            f"If start_pose is specified it must be a 1D array of length 3. Instead it is of shape {start_pose.shape}"

    # If d_odom has 2 columns add a column of zeros in the middle (for dy)
    if d_odom.shape[1] == 2:
        tmp = np.zeros((len(d_odom), 3))
        tmp[:, [0, 2]] = d_odom
        d_odom = tmp

    # We expect dt to be a 1D array. It may be a (n, 1) array, in which case we'll reshape it to (n,).
    dt = dt.squeeze()


    # Unroll thetas first. This is because dx, dy are dependent on value of theta at each step.
    thetas = np.cumsum(d_odom[:, 2] * dt, axis=0) + start_pose[2]

    # Create vectors along and orthogonal to theta
    along_vec = np.concatenate((np.cos(thetas)[..., None], np.sin(thetas)[..., None]), axis=1)
    # Orthogonal vector is -sin(theta) along x and cos(theta) along y, so we can just use along
    ortho_vec = along_vec[..., [1, 0]]
    ortho_vec[..., 0] *= -1

    # Unroll the poses
    poses = start_pose[:2] + np.cumsum(dt[..., None] * (along_vec * d_odom[:, 0, None] + ortho_vec * d_odom[:, 1, None]), axis=0)

    result = np.hstack((poses, thetas[:, None]))
    result = np.vstack((start_pose[None, :], result))

    return result


def reconstruct_poses_from_acc(acc: np.ndarray, dt: np.ndarray, start_pose: Optional[np.ndarray] = None, start_vel: Optional[np.ndarray] = None):
    if start_pose is None:
        start_pose = np.zeros(3)
    if start_vel is None:
        start_vel = np.zeros(3)

    # Convert from (x'', theta'') to (x'', y'', theta'')
    if acc.shape[1] == 2:
        tmp = np.zeros((len(acc), 3))
        tmp[:, [0, 2]] = acc
        acc = tmp

    # dt can sometimes be (n, 1), so just for sanity check we'll make sure it's (n,).
    dt = dt.squeeze()

    # 1. Rollout thetas
    disp_v_theta = np.cumsum(acc[:, 2] * dt)
    theta_cum = np.cumsum((start_vel[2] + disp_v_theta) * dt)
    thetas = (theta_cum + start_pose[2]).squeeze()

    # 2. Rollout velocities
    along_vec = np.concatenate((np.cos(thetas)[..., None], np.sin(thetas)[..., None]), axis=1)
    ortho_vec = along_vec[..., [1, 0]]

    # 3. Rollout velocities
    disp_v = np.cumsum(dt[..., None] * (along_vec * acc[:, 0, None] + ortho_vec * acc[:, 1, None]), axis=0) + start_vel[None, :2]
    disp_v = np.hstack((disp_v, disp_v_theta[:, None]))

    v = disp_v + start_vel

    return np.cumsum(dt[..., None] * v, axis=0) + start_pose
