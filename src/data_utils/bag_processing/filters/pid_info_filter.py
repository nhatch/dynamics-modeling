from typing import List, Set

from rospy import Time

from ..msg_stubs import PIDInfo, PolarisControlMode
from .abstract_filter import AbstractFilter


class PIDInfoFilter(AbstractFilter):
    name = "pid_info"
    topics = [{'/{robot_name}/pid_info'}]

    def __init__(self) -> None:
        super().__init__()

        # Initialize per-bag variables
        self.end_bag()

    @property
    def should_log(self) -> bool:
        return self.cur_state

    def callback(self, msg: PIDInfo, ts: Time, topic: str):

        self.cur_state = (
            msg.polaris_control_mode == PolarisControlMode.AUTONOMOUS.value
            and msg.polaris_control_health == 2
            and msg.brake_responding  # If brake is not responding then data will not be reliable, so it's good to filter this too
        )

    def end_bag(self):
        self.cur_state = False