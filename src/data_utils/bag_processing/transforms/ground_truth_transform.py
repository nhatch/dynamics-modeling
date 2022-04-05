from typing import List, Tuple
import numpy as np
import rospy
from .abstract_transform import AbstractTransform
from nav_msgs.msg import Odometry
import scipy.spatial.transform as trf


class GroundTruthTransform(AbstractTransform):
    topics = ["/unity_command/ground_truth/{robot_name}", "/{robot_name}/odom"]
    feature = "target"

    def __init__(self, features: List[str]):
        super().__init__(features)

        self.previous_pose = None

        self.end_bag()

    def callback(self, msg: Odometry, ts: rospy.Time, current_state, *args, **kwargs):
        # z_angle = trf.Rotation([
        #     msg.pose.pose.orientation.x,
        #     msg.pose.pose.orientation.y,
        #     msg.pose.pose.orientation.z,
        #     msg.pose.pose.orientation.w,
        # ]).as_euler("zyx")[0]
        # state = [msg.pose.pose.position.x, msg.pose.pose.position.y, z_angle]

        state = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        self.state_history.append(state)
        self.ts_history.append(ts.to_sec())

        current_state[self.__class__.feature] = np.array(state)

    def end_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        states = np.array(self.state_history)
        ts = np.array(self.ts_history)

        self.end_bag()

        return states, ts

    def end_bag(self):
        self.state_history = []
        self.ts_history = []
