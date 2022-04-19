from typing import List

import numpy as np
import rospy
import scipy.spatial.transform as trf
from nav_msgs.msg import Odometry
from scipy import interpolate

from .abstract_transform import AbstractTransform

# TODO: Finish this. It should use derivatives for speed etc.


class AutorallyGroundTruth(AbstractTransform):
    topics = ["/{robot_name}/odom"]
    feature = "autorally-ground-truth"

    def __init__(self, features: List[str], use_quarterions: bool = True):
        """Ground Truth based on Autorally project."""
        super().__init__(features)

        self.previous_pose = None
        self.use_quarterions = use_quarterions

        self.history_poses: List[List[float]] = []
        self.history_ts: List[float] = []

        self.spline_pwr = 3

    def callback(self, msg: Odometry, ts: rospy.Time, current_state, *args, **kwargs):
        # Get angle of pose
        angle = np.array(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )
        if self.use_quarterions:
            angle = trf.Rotation(angle).as_euler("xyz")
            ya = angle[-1]
        else:
            ya = trf.Rotation(angle).as_euler("xyz")[-1]

        # Get velocity
        rot_mat = np.array([[np.cos(ya), np.sin(ya)], [-np.sin(ya), np.cos(ya)]])
        vel_wf = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        vel_bf = np.dot(rot_mat, vel_wf)
        vel_x = vel_bf[0]
        vel_y = vel_bf[1]

        heading_rate = msg.twist.twist.angular.z

        pose = [*angle, vel_x, vel_y, heading_rate]
        self.history_poses.append(pose)
        self.history_ts.append(ts.to_sec())

        current_state[self.__class__.feature] = np.array(pose)

    def end_sequence(self):
        poses = np.array(self.history_poses)
        ts = np.array(self.history_ts)

        # Move absolute timestamps to relative time
        # NOTE: Is this easier for interpolation? Autorally uses relative time, so I use it too.
        start_time = ts[0]
        ts -= start_time

        # NOTE: What are knots used for?
        knots = np.linspace(ts[0], ts[-1], int(round(ts.size / 10.0)))[1:-1]

        # Interpolate
        # splrep and splev works for 1D arrays only, so we need to apply it per-column
        for f_idx in range(poses.shape[1]):
            spline_params = interpolate.splrep(
                ts, poses[:, f_idx], k=self.spline_pwr, t=knots
            )

            poses[:, f_idx] = interpolate.splev(ts, spline_params)

        # Reset poses/ts for next sequence
        self.end_bag()

        # Add relative time back to absolute time
        return poses, ts + start_time

    def end_bag(self):
        # Reset history
        self.history_poses = []
        self.history_ts = []
