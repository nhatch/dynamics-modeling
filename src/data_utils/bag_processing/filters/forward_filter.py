from typing import Union
from genpy import Duration
from rospy import AnyMsg, Time
from nav_msgs.msg import Odometry

from data_utils.bag_processing.msg_stubs import PIDInfo
from .abstract_filter import AbstractFilter

class ForwardFilter(AbstractFilter):
    name = "forward"
    topics = [{'/{robot_name}/pid_info', '/{robot_name}/odom'}]

    def __init__(self) -> None:
        super().__init__()

        # Initialize per-bag variables
        self.end_bag()

        self.after_last_threshold_log = Duration(1)  # seconds

    @property
    def should_log(self) -> bool:
        return self.cur_state

    def callback(self, msg: Union[Odometry, PIDInfo], ts: Time, topic: str):
        if topic.endswith("odom"):
            msg: Odometry
            vel = msg.twist.twist.linear.x
        else:
            msg: PIDInfo
            vel = msg.vel
        self.cur_state = (
            vel > 1e-6 or (ts - self.last_forward_msg) < self.after_last_threshold_log
        )

        if vel > 1e-6:
            self.last_forward_msg = ts

    def end_bag(self):
        self.cur_state = False
        self.last_forward_msg = Time()