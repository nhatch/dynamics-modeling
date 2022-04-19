from typing import Dict, Tuple

import numpy as np
import rospy

from ..msg_stubs import AutomaticGearDirectionDict, VehicleInput
from .abstract_transform import AbstractTransform


class InputTransform(AbstractTransform):
    topics = ["/{robot_name}/input"]
    feature = "control"

    THRESHOLD_NEW_SEQUENCE = (
        0.15  # seconds. Time after which a new sequence is started.
    )

    def __init__(self, features):
        super().__init__(features)
        self.required_features_set = set(
            self.required_features
        )  # This is to speed up check in the beggining of callback. It will run often

        self.end_bag()

    def callback(
        self,
        msg: VehicleInput,
        ts: rospy.Time,
        current_state: Dict[str, np.ndarray],
        *args,
        **kwargs,
    ):
        state = [
            msg.steer * AutomaticGearDirectionDict[msg.automatic_gear],
            msg.throttle * AutomaticGearDirectionDict[msg.automatic_gear],
            msg.brake,
        ]

        # Add vehicle input information to the current state.
        current_state[self.__class__.feature] = np.array(state)

        self.state_history.append(state)
        self.ts_history.append(ts.to_sec())

    def end_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        states = np.array(self.state_history)
        ts = np.array(self.ts_history)

        self.end_bag()

        return states, ts

    def end_bag(self):
        self.state_history = []
        self.ts_history = []
