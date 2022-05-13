from rospy import Time

from ..msg_stubs import PIDInfo, PolarisControlMode
from .abstract_filter import AbstractFilter


class PIDInfoFilter(AbstractFilter):
    topics = [{"/{robot_name}/pid_info"}]

    def __init__(self) -> None:
        super().__init__()

        # Initialize per-bag variables
        self.end_bag()

    @property
    def should_log(self) -> bool:
        return self.cur_state

    def callback(self, msg: PIDInfo, ts: Time, topic: str):

        # There are two versions of PIDInfo msg.
        if hasattr(msg, "brake_responding"):
            self.cur_state = (
                msg.polaris_control_mode == PolarisControlMode.AUTONOMOUS.value
                and msg.polaris_control_health == 2
                # If brake/throttle is not responding then data will not be reliable, so it's good to filter this too
                and msg.brake_responding
            )
        else:
            self.cur_state = (
                msg.polaris_control_mode == PolarisControlMode.AUTONOMOUS.value
                and msg.polaris_control_health == 2
                # If brake/throttle is not responding then data will not be reliable, so it's good to filter this too
                and not msg.brake_not_responding
                and not msg.throttle_not_responding
            )

    def end_bag(self):
        self.cur_state = False
