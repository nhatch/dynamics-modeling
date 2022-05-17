from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
import rospy

from ..msg_stubs import AutomaticGearDirectionDict, VehicleInput, VehicleState
from .abstract_transform import AbstractTransform


class InputTransform(AbstractTransform):
    topics = [{"/{robot_name}/input", "/{robot_name}/vehicle"}]
    feature = "control"

    THRESHOLD_NEW_SEQUENCE = (
        0.15  # seconds. Time after which a new sequence is started.
    )

    def __init__(self):
        super().__init__()
        self.__last_known_gear: Optional[int] = None

        self.end_bag()

    def callback(
        self,
        topic: str,
        msg: Union[VehicleInput, VehicleState],
        ts: rospy.Time,
        current_state: Dict[str, np.ndarray],
        *args,
        **kwargs,
    ):
        if topic.endswith("input"):
            msg = cast(VehicleInput, msg)

            if self.__last_known_gear is None:
                return

            state = [
                # msg.throttle * AutomaticGearDirectionDict[self.__last_known_gear],
                msg.throttle,
                msg.brake,
                msg.steer,
            ]

            # Add vehicle input information to the current state.
            current_state[self.__class__.feature] = np.array(state)

            self.state_history.append(state)
            self.ts_history.append(ts.to_sec())
        elif topic.endswith("vehicle"):
            msg = cast(VehicleState, msg)
            self.__last_known_gear = msg.gearbox_mode

    def end_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        states = np.array(self.state_history)
        ts = np.array(self.ts_history)

        self.end_bag()

        return states, ts

    def end_bag(self):
        self.state_history = []
        self.ts_history = []
