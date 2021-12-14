from typing import List, Union
import rosbag
from pathlib import Path
import numpy as np
from .general_utils import planar_pose, planar_twist


MODE_TO_DIRECTION_DICT = {
    0: 1.0,  # Manual
    1: 0.0,  # Park
    2: -1.0,  # Rear
    3: 0.0,  # Neutral
    4: 1.0,  # Drive
    5: 1.0,  # Low Drive
}

def compare_input_msgs(a, b):
    fields = ["steer", "brake", "throttle", "automatic_gear"]
    return all([getattr(a, x) == getattr(b, x) for x in fields])


def process_raw_sequence(seq_raw: List[List[float]]) -> np.array:
    seq_processed = np.zeros((len(seq_raw), 3 + 2 + 6))

    # Copy Control & State
    seq_np = np.array(seq_raw)
    seq_processed[:, :5] = seq_np[:, :5]

    # Convert GT to planar pose/twist
    p_pose = planar_pose(seq_np[:, 5:9])  # poses
    p_twist = planar_twist(seq_np[:, 9:12], p_pose[2], seq_np[:, 12])
    seq_processed[:, 5:8] = p_pose.T
    seq_processed[:, 8:11] = p_twist.T

    return seq_processed


def bag_extract_data(dataset_name: str, bag_file_path: Union[str, Path]):
    bag = rosbag.Bag(bag_file_path)
    topics = bag.get_type_and_topic_info().topics

    # Get robot name. It should be the most often occuring topmost key
    topic_parents, topic_parents_counts =  np.unique([x.split("/", 2)[1] for x in topics.keys()], return_counts=True)
    robot_name = topic_parents[np.argmax(topic_parents_counts)]

    # TODO: Detect in input/state pair is here.
    # This changes a flow significantly, in which case should we sync D and H first.

    # Input (D - control)
    input_topic = f"/{robot_name}/input"

    # Odom (H - state)
    state_topic = f"/{robot_name}/odom"  # TODO: Could also be f"/{robot_name}/vehicle/odom"

    # Odom (P - ground truth)
    # TODO: Verify what is ground truth in this case. Jakub: Possible candidates seem to be /pose, /odom, /vehicle/odom
    gt_topic = f"/unity_command/ground_truth/{robot_name}"  # Sim
    if gt_topic not in topics:
        # TODO: Could also be f"/{robot_name}/vehicle/odom"
        gt_topic = f"/{robot_name}/odom"

    topics_to_extract = [input_topic, state_topic, gt_topic]
    for topic in topics_to_extract:
        if state_topic not in topics:
            raise ValueError(
                f"Topic: {state_topic} not found in the bag. Stopping..."
            )

    # Process data
    result = []
    seq_raw = []  # Current sequence

    # Input
    last_input_state_pair_to_log = []
    last_input_ts = 0
    last_state_message = None
    last_gt_message = None
    for topic, msg, ts in bag.read_messages(topics=[input_topic, state_topic, gt_topic]):
        if topic == input_topic:
            # TODO: Check if the input is the same as one before. If so continue.
            # Real vehicle spams /input message quite often (avg./median ~30 Hz, range 25-40Hz)
            # These seem to be quite different at every step. It might be that PID is modifying the steps.
            # It would be really beneficial to not have such an overlapping data.
            # However, this make spacing of commands inconsistent across time, which can cause issues.
            if last_input_state_pair_to_log:
                input_msg, state_msg = last_input_state_pair_to_log
                seq_raw.append(
                    [
                        # Control
                        input_msg.steer * MODE_TO_DIRECTION_DICT[input_msg.automatic_gear],
                        input_msg.throttle * MODE_TO_DIRECTION_DICT[input_msg.automatic_gear],
                        input_msg.brake,
                        # State
                        state_msg.twist.twist.linear.x,
                        state_msg.twist.twist.angular.z,
                        # Ground Truth
                        # GT - Will be converted to planar pose
                        last_gt_message.pose.pose.orientation.x,
                        last_gt_message.pose.pose.orientation.y,
                        last_gt_message.pose.pose.orientation.z,
                        last_gt_message.pose.pose.orientation.w,
                        # GT - Will be converted to planar twist
                        last_gt_message.twist.twist.linear.x,
                        last_gt_message.twist.twist.linear.y,
                        last_gt_message.twist.twist.linear.z,
                        last_gt_message.twist.twist.angular.z,
                    ]
                )

                # Check if sequence should be closed (Diff > 0.15s)
                # TODO: Make an argument
                if (ts - last_input_ts).to_sec() > 0.15:
                    # Add sequence to result
                    result.append(process_raw_sequence(seq_raw))
                    seq_raw = []

            # Save current ts/msg for next iter
            if last_state_message is not None:
                last_input_state_pair_to_log = [msg, last_state_message]
                last_input_ts = ts

        if topic == state_topic:
            last_state_message = msg
        if topic == gt_topic:
            last_gt_message = msg

    # Fence
    if seq_raw:
        result.append(process_raw_sequence(seq_raw))

    bag.close()
    return result
