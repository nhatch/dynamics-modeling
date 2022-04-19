from typing import List, Tuple

import numpy as np
import rospy
from nav_msgs.msg import Odometry

from .abstract_transform import AbstractTransform


class OdomTransform(AbstractTransform):
    topics = ["/{robot_name}/odom"]
    feature = "state"

    def __init__(self, features: List[str]):
        super().__init__(features)
        self.end_bag()

    def callback(self, msg: Odometry, ts: rospy.Time, current_state, *args, **kwargs):
        # Dictionaries are modified in place in python
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
