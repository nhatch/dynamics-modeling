from typing import Tuple

import numpy as np
import rospy
from nav_msgs.msg import Odometry

from .abstract_transform import AbstractTransform


class OdomTransform(AbstractTransform):
    topics = [{"/{robot_name}/odom"}]
    feature = "state"

    def __init__(self):
        super().__init__()
        self.end_bag()

    def callback(
        self, topic: str, msg: Odometry, ts: rospy.Time, current_state, *args, **kwargs
    ):
        state = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        self.state_history.append(state)
        self.ts_history.append(ts.to_sec())

        # Dictionaries are modified in place in python
        current_state[self.__class__.feature] = np.array(state)

    def end_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        states = np.array(self.state_history)
        ts = np.array(self.ts_history)

        self.end_bag()

        return states, ts

    def end_bag(self):
        self.state_history = []
        self.ts_history = []
