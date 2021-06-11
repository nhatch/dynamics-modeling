#!/home/nhatch2/miniconda3/bin/python3

import csv
import sys
import numpy as np
import scipy.spatial.transform as trf

if len(sys.argv) < 2:
    print("You need to specify DATASET_NAME")
    sys.exit(0)

target = sys.argv[1]
cmd_vel_file = open('datasets/' + target + '/cmd_vel.txt', newline='')
odom_file = open('datasets/' + target + '/odom.txt', newline='')
gt_pose_file = open('datasets/' + target + '/gt_pose.txt', newline='')
gt_twist_file = open('datasets/' + target + '/gt_twist.txt', newline='')

cmd_vel_reader = csv.DictReader(cmd_vel_file)
odom_reader = csv.DictReader(odom_file)
gt_pose_reader = csv.DictReader(gt_pose_file)
gt_twist_reader = csv.DictReader(gt_twist_file)

data = []

prev_cmd_vel_time = 0

def pose_difference(p2, p1):
    r1 = trf.Rotation.from_quat(p1[3:])
    r2 = trf.Rotation.from_quat(p2[3:])
    world_trans = (p2-p1)[:3]
    trans_vs_p1 = r1.inv().apply(world_trans)
    # TODO not 100% sure how to verify this is "correct" or whether this is really
    # the format that we want our dynamics models to predict
    rot_vs_p1 = (r1.inv() * r2).as_euler('zyx')
    return np.concatenate([trans_vs_p1, rot_vs_p1])

def xyt_pose(pose):
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

def xz_twist(twist, pose):
    # We need the `pose` because `twist` is given in the world frame,
    # while we want it in the robot base frame.
    #
    # HOWEVER, making this transform is hard, and actually I don't think
    # I necessarily need it for my dynamics models (at least not until
    # we get to recurrent models).
    return np.array([])

for row in cmd_vel_reader:
    cmd_vel_time = int(row['%time'])

    cmd_vel = get_fields(row, cmd_vel_fields)
    # There are many GT pose readings for each cmd_vel, so let's not
    # worry about interpolating to try to synchronize these timestamps.
    gt_pose = first_after(cmd_vel_time, gt_pose_fields, gt_pose_reader)
    gt_twist = first_after(cmd_vel_time, gt_twist_fields, gt_twist_reader)
    # Same for odom
    odom_twist = first_after(cmd_vel_time, odom_twist_fields, odom_reader)

    # sanity check this was 100ms
    time_diff = (cmd_vel_time - prev_cmd_vel_time) // 1000000
    if time_diff == 0:
        continue
    elif time_diff != 100:
        print("WARNING: Weird time diff", time_diff)
        seq_no += 1

    data_row = np.concatenate([[seq_no], cmd_vel, odom_twist,
            xyt_pose(gt_pose), xz_twist(gt_twist, gt_pose)])
    data.append(data_row)

    prev_cmd_vel_time = cmd_vel_time

np.savetxt('datasets/' + target + '/np_v1.txt', data)

cmd_vel_file.close()
odom_file.close()
gt_pose_file.close()
gt_twist_file.close()
