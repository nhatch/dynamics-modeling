from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from nav_msgs.msg import Odometry
import rospy
from scipy.spatial.transform import Rotation
import tqdm
import rosbag


def unroll_sequence_with_ts(twists: np.ndarray, ts: np.ndarray):
    start_state = np.zeros(twists.shape[1])

    seq = np.zeros((len(twists)+1, twists.shape[1]))
    seq[0] = start_state
    for t in range(len(twists)):
        curr_angle = seq[t,2]
        summand = twists[t,:]
        assert(summand.shape == (3,))
        world_summand = np.zeros_like(summand)
        world_summand[0] = np.cos(curr_angle) * summand[0] - np.sin(curr_angle) * summand[1]
        world_summand[1] = np.sin(curr_angle) * summand[0] + np.cos(curr_angle) * summand[1]
        world_summand[2] = summand[2]
        seq[t+1,:] = seq[t,:] + world_summand
    return seq


def plot_arr(axis: Axes, data: np.ndarray, t: np.ndarray, label: str = "", *args, **kwargs):
    poses = reconstruct_from_odoms(data, t, *args, **kwargs)

    axis.scatter(poses[:, 0], poses[:, 1], alpha=0.1)
    # axis.plot(seq[:, 0], seq[:, 1])
    # axis.plot(np.cumsum(t), data[:, 2])
    # print(np.cumsum(t)[-1])
    # print(t[:5])
    axis.set_title(label)


def parse_odom_msgs(msgs: List[Odometry], ts: List[rospy.Time], delay: int = 0, n_steps: int = 1):
    back_idx_offset = delay + n_steps
    assert back_idx_offset >= 1

    assert len(msgs) == len(ts)

    time_diff = []
    poses = []

    vecs_along = []
    vecs_ortho = []

    for i in range(back_idx_offset, len(msgs)):

        prev_msg = msgs[i - back_idx_offset]
        prev_ts = ts[i - back_idx_offset]
        cur_msg = msgs[i]
        cur_ts = ts[i]

        # Time
        time_diff.append((cur_ts - prev_ts).to_sec())

        # Heading at last angle
        z_angle = Rotation(np.array([
            prev_msg.pose.pose.orientation.x,
            prev_msg.pose.pose.orientation.y,
            prev_msg.pose.pose.orientation.z,
            prev_msg.pose.pose.orientation.w,
        ])).as_euler('zyx')[0]
        heading_vec = np.array([1., np.tan(z_angle)])
        heading_vec /= np.linalg.norm(heading_vec)

        ortho_vec = heading_vec[[1, 0]]
        ortho_vec[0] *= -1

        # Get vector of displacement
        disp = np.array([
            cur_msg.pose.pose.position.x - prev_msg.pose.pose.position.x,
            cur_msg.pose.pose.position.y - prev_msg.pose.pose.position.y,
        ])

        # project
        proj_x = (heading_vec @ disp) * heading_vec
        proj_y = (ortho_vec @ disp) * ortho_vec

        # To get values for x and y dot projection with heading and ortho vectors
        proj_x = proj_x @ heading_vec
        proj_y = proj_y @ ortho_vec

        # Lastly get change in theta
        cur_z_angle = Rotation(np.array([
            cur_msg.pose.pose.orientation.x,
            cur_msg.pose.pose.orientation.y,
            cur_msg.pose.pose.orientation.z,
            cur_msg.pose.pose.orientation.w,
        ])).as_euler('zyx')[0]
        proj_theta = cur_z_angle - z_angle

        poses.append([proj_x, proj_y, proj_theta])

        vecs_along.append(heading_vec)
        vecs_ortho.append(ortho_vec)

    return np.array(poses), np.array(time_diff), np.array(vecs_along), np.array(vecs_ortho)


def reconstruct_from_odoms(odoms: np.ndarray, tsps: np.ndarray, start_pose: Optional[np.ndarray] = None, delay: int = 0, n_steps: int = 1):
    back_idx_offset = delay + n_steps

    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])

    # Rollout each continuous sequence seperately
    # Continuos sequence is meant by markovian chain where theta(i) = d_theta(i, j) + theta(j)
    size_rollout = len(odoms) // back_idx_offset
    thetas = np.reshape(odoms[:size_rollout * back_idx_offset, 2], (size_rollout, back_idx_offset))
    thetas = np.cumsum(thetas, axis=0) + start_pose[2]

    along_vec = np.ones(list(thetas.shape) + [2])
    along_vec[..., 1] = np.tan(thetas)
    along_vec /= np.linalg.norm(along_vec, axis=2, keepdims=True)

    ortho_vec = along_vec[..., [1, 0]]
    ortho_vec[..., 0] *= -1

    along_vals = np.reshape(odoms[:size_rollout * back_idx_offset, 0], (size_rollout, back_idx_offset))
    ortho_vals = np.reshape(odoms[:size_rollout * back_idx_offset, 1], (size_rollout, back_idx_offset))

    poses = np.cumsum(along_vec * along_vals[:, None] + ortho_vec * ortho_vals[:, None], axis=0)

    poses = np.transpose(poses, (1, 0, 2)).reshape(-1, 2)
    poses += start_pose[:2]

    return poses


def main():

    raw_pred = np.load("data/y_pred.npy")
    raw_true = np.load("data/y_true.npy")
    ts = np.load("data/ts.npy", allow_pickle=True)
    # print(ts)

    # print("Here")
    bag = rosbag.Bag("datasets/rzr_real/offtrail_7.5ms_OFPWS_6000_nv_ecc_reverse_2022-02-12-00-19-10.bag")
    plt_x = []
    plt_y = []
    plt_theta = []

    odom_msgs = []
    tsps = []
    for topic, msg, ts in tqdm.tqdm(bag.read_messages("/crl_rzr/odom")):
        # print(f"position: {msg.pose.pose.position.x}")
        # print(f"orientation: {msg.pose.pose.orientation.x}")
        odom_msgs.append(msg)
        tsps.append(ts)
        plt_x.append(msg.pose.pose.position.x)
        plt_y.append(msg.pose.pose.position.y)
        z_angle = Rotation(np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])).as_euler('zyx')[0]
        plt_theta.append(z_angle)

    fig, ax = plt.subplots(1, 2)
    plot_arr(ax[0], raw_pred, ts, "Pred", delay=0, n_steps=1, start_pose=np.array([0, 0, plt_theta[0]]),)
    plot_arr(ax[1], raw_true, ts, "True", delay=0, n_steps=1, start_pose=np.array([0, 0, plt_theta[0]]),)
    plt.show()


    DELAY_STEPS = 1
    N_STEPS = 1

    odoms, ts, along_vecs, ortho_vecs = parse_odom_msgs(odom_msgs, tsps, delay=DELAY_STEPS, n_steps=N_STEPS)
    start_pose = np.array(
        [
            plt_x[0],
            plt_y[0],
            plt_theta[0],
        ]
    )
    poses = reconstruct_from_odoms(odoms, ts, np.array([0, 0, plt_theta[0]]), delay=DELAY_STEPS, n_steps=N_STEPS)

    fig, ax = plt.subplots(1, 2)
    # ax[0].scatter(poses[:, 0], poses[:, 1], alpha=0.1)
    # # ax[0].plot(np.cumsum(odoms[:, 2]) + start_pose[2])
    # ax[0].set_title("Transformed")
    # ax[1].scatter(plt_x, plt_y, alpha=0.1)
    # # ax[1].plot(plt_theta)
    # ax[1].set_title("Original")

    # for t_x, t_y, pose, along, ortho in zip(plt_x[::(DELAY_STEPS + N_STEPS)], plt_y[::(DELAY_STEPS + N_STEPS)], poses, along_vecs, ortho_vecs):
    for t_x, t_y, pose, along, ortho in zip(plt_x[:len(poses)], plt_y[:len(poses)], poses, along_vecs, ortho_vecs):
        inner_prod = along @ ortho
        # print(f"along scale: {along @ along}")
        # print(f"ortho scale: {ortho @ ortho}")
        if inner_prod > 0:
            print(f"Inner product: {inner_prod}")

        ax[0].arrow(t_x, t_y, ortho[0], ortho[1], color="orange")
        ax[0].arrow(t_x, t_y, along[0], along[1], color="blue")

        ax[1].arrow(pose[0], pose[1], ortho[0], ortho[1], color="orange")
        ax[1].arrow(pose[0], pose[1], along[0], along[1], color="blue")

    # plt.scatter(plt_x, plt_y, alpha=0.1)
    plt.show()


def hash_arr(arr: np.ndarray):
    prev_writeable_state = arr.flags.writeable
    arr.flags.writeable = False

    result = []

    for a in arr:
        result.append(hash(a.tobytes()))
    arr.flags.writeable = prev_writeable_state
    return result


def main2():
    y = np.load("data/lookahead_y.npy")

    bag = rosbag.Bag("datasets/rzr_real/offtrail_7.5ms_OFPWS_6000_nv_ecc_reverse_2022-02-12-00-19-10.bag")

    y_bag = []

    for topic, msg, ts in tqdm.tqdm(bag.read_messages("/crl_rzr/odom")):
        z_angle = Rotation(np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])).as_euler('zyx')[0]
        y_bag.append(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, z_angle]))
    y_bag = np.array(y_bag)

    y_bag_set = set(hash_arr(y_bag))
    y_set = set(hash_arr(y))

    print(f"In bag - in dataset: {len(y_bag_set - y_set)}")
    print(f"in dataset - In bag: {len(y_set - y_bag_set)}")

    diff = y[1:] - y[:-1]
    print(f"Duplicates: {np.sum(np.all(diff == 0, axis=1))}")
    print(f"Size bag: {len(y_bag)}")
    print(f"Size dataset: {len(y)}")
    plt.scatter(diff[:, 0], diff[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
    # main2()
