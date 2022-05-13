from typing import Union, cast

from nav_msgs.msg import Odometry
from rospy import Duration, Time

from ..msg_stubs import PIDInfo
from .abstract_filter import AbstractFilter


class BackwardFilter(AbstractFilter):
    topics = [{"/{robot_name}/pid_info", "/{robot_name}/odom"}]

    def __init__(self) -> None:
        super().__init__()

        # Initialize per-bag variables
        self.end_bag()

        self.after_last_threshold_log = Duration(1)  # seconds
        self.cur_state = False
        self.last_forward_msg = Time()

    @property
    def should_log(self) -> bool:
        return self.cur_state

    def callback(self, msg: Union[Odometry, PIDInfo], ts: Time, topic: str):
        if topic.endswith("odom"):
            msg = cast(Odometry, msg)
            vel = msg.twist.twist.linear.x
        else:
            msg = cast(PIDInfo, msg)
            vel = msg.vel

        # Compare to less than -1e-6 since ROS has often floating point errors
        is_vel_negative = vel < -1e-6

        self.cur_state = (
            is_vel_negative
            or (ts - self.last_forward_msg) < self.after_last_threshold_log
        )

        if is_vel_negative:
            self.last_forward_msg = ts

    def end_bag(self):
        self.cur_state = False
        self.last_forward_msg = Time()
