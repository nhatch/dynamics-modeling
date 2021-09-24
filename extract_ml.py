#!/usr/bin/env python3

import csv
import sys
import numpy as np
import scipy.spatial.transform as trf

if len(sys.argv) < 2:
    print("You need to specify DATASET_NAME")
    sys.exit(0)
target = sys.argv[1]

is_ackermann = False
try:
    cmd_vel_file = open('datasets/' + target + '/cmd_vel.txt', newline='')
    control_reader = csv.DictReader(cmd_vel_file)
except FileNotFoundError:
    is_ackermann = True
    vinput_file = open('datasets/' + target + '/input.txt', newline='')
    control_reader = csv.DictReader(vinput_file)

gt_pose_file = open('datasets/' + target + '/gt_pose.txt', newline='')
gt_pose_reader = csv.DictReader(gt_pose_file)

has_twist = True
try:
    odom_file = open('datasets/' + target + '/odom.txt', newline='')
    gt_twist_file = open('datasets/' + target + '/gt_twist.txt', newline='')
    odom_reader = csv.DictReader(odom_file)
    gt_twist_reader = csv.DictReader(gt_twist_file)
except FileNotFoundError:
    has_twist = False;

data = []

prev_control_time = 0

def pose_difference(p2, p1):
    r1 = trf.Rotation.from_quat(p1[3:])
    r2 = trf.Rotation.from_quat(p2[3:])
    world_trans = (p2-p1)[:3]
    trans_vs_p1 = r1.inv().apply(world_trans)
    # TODO not 100% sure how to verify this is "correct" or whether this is really
    # the format that we want our dynamics models to predict
    rot_vs_p1 = (r1.inv() * r2).as_euler('zyx')
    return np.concatenate([trans_vs_p1, rot_vs_p1])

def planar_pose(pose):
    r1 = trf.Rotation.from_quat(pose[3:])
    rot = r1.as_euler('zyx')
    return np.array([pose[0], pose[1], rot[0]])

def first_after(query_time, fields, reader):
    for row in reader:
        row_time = int(row['field.header.stamp'])
        if row_time > query_time:
            return get_fields(row, fields)

def get_fields(row, fields):
    r = []
    for field in fields:
        r.append(float(row[field]))
    return np.array(r)

seq_no = 0

cmd_vel_fields = [
        'field.linear.x',
        'field.angular.z']

vinput_fields = [
        'field.steer',
        'field.throttle',
        'field.brake',
        'field.automatic_gear']

# These are in the robot base frame, not the world frame.
odom_twist_fields = [
        'field.twist.twist.linear.x',
        'field.twist.twist.angular.z']

gt_pose_fields = [
        'field.pose.position.x',
        'field.pose.position.y',
        'field.pose.position.z',
        'field.pose.orientation.x',
        'field.pose.orientation.y',
        'field.pose.orientation.z',
        'field.pose.orientation.w']

# These are in the world frame, not the robot base frame.
gt_twist_fields = [
        'field.twist.linear.x',
        'field.twist.linear.y',
        'field.twist.linear.z',
        'field.twist.angular.x',
        'field.twist.angular.y',
        'field.twist.angular.z']

def planar_twist(twist, pose):
    # We need the `pose` because `twist` is given in the world frame,
    # while we want it in the robot base frame.
    theta = planar_pose(pose)[2]
    linear_vel = np.cos(theta) * twist[0] + np.sin(theta) * twist[1]
    # Transverse vel should be zero except for projection errors and
    # times when the robot is slipping sideways.
    transverse_vel = -np.sin(theta) * twist[0] + np.cos(theta) * twist[1]
    return np.array([linear_vel, transverse_vel, twist[5]])

num_zero = 0
total_seq_time = 0
total_seq_steps = 0
in_brake_phase = False
for row in control_reader:
    control_time = int(row['%time'])
    if is_ackermann:
        control_raw = get_fields(row, vinput_fields)
        control = control_raw[:3]
        if control_raw[3] == 2:
            # TODO this way of handling forward/reverse assumes that whenever
            # our forward velocity is positive, we are in "drive"; and whenever
            # negative, we're in "reverse". Is that assumption valid? E.g. when
            # slipping downhill? It might not matter if we're not turning.
            control *= -1
        elif control_raw[3] != 4:
            print("ERROR: Control gear {} was not 2 or 4".format(int(control_raw[3])))
            sys.exit(1)
        if control_raw[2] == 0.5 and control_raw[0] == 0.0:
            # We're in the braking phase at the end of the sequence
            in_brake_phase = True
            continue
        elif in_brake_phase:
                print("Average step length for this seq: {:.3f}".format(
                    total_seq_time / total_seq_steps))
                in_brake_phase = False
                seq_no += 1
                total_seq_time = 0
                total_seq_steps = 0
    else:
        control = get_fields(row, cmd_vel_fields)

    # sanity check this was 100ms
    time_diff = (control_time - prev_control_time) // 1000000
    if time_diff == 0:
        print("WARNING: Got time diff of zero, "
            "control {:.3f} {:.3f} vs previous {:.3f} {:.3f}".format(
            control[0], control[1], data[-1][1], data[-1][2]))
        num_zero += 1
        # Let's assume whichever was later in the rosbag is the one that
        # actually got executed on the system.
        # TODO: Why is this happening? Should I collect cmd_vel_stamped?
        data[-1][1] = control[0]
        data[-1][2] = control[1]
        continue
    # For some reason the time diffs for the RZR are all over the place
    # TODO figure out why.
    elif (time_diff != 100 and not is_ackermann) or (time_diff > 200):
        print("WARNING: Weird time diff", time_diff)
        seq_no += 1
    else:
        total_seq_steps += 1
        total_seq_time += time_diff

    # There are many GT pose readings for each control, so let's not
    # worry about interpolating to try to synchronize these timestamps.
    gt_pose = first_after(control_time, gt_pose_fields, gt_pose_reader)

    odom_twist = np.array([])
    planar_gt_twist = np.array([])
    if has_twist:
        gt_twist = first_after(control_time, gt_twist_fields, gt_twist_reader)
        planar_gt_twist = planar_twist(gt_twist, gt_pose)
        odom_twist = first_after(control_time, odom_twist_fields, odom_reader)

    # TODO we need to do non-planar pose / twist. (Maybe in addition to planar,
    # so we can do ablation testing.)
    data_row = np.concatenate([[seq_no], control, odom_twist,
            planar_pose(gt_pose), planar_gt_twist])
    data.append(data_row)

    prev_control_time = control_time

print("Got num zero:", num_zero)
np.savetxt('datasets/' + target + '/np.txt', data)

if is_ackermann:
    vinput_file.close()
else:
    cmd_vel_file.close()
gt_pose_file.close()
if has_twist:
    odom_file.close()
    gt_twist_file.close()
