#!/home/nhatch2/miniconda3/bin/python3

import csv
import numpy as np
import scipy.spatial.transform as trf

target = 'sim_data'
cmd_vel_file = open(target + '/cmd_vel.txt', newline='')
pose_file = open(target + '/pose.txt', newline='') # ground truth

cmd_vel_reader = csv.DictReader(cmd_vel_file)
pose_reader = csv.DictReader(pose_file)

data = []

prev_cmd_vel_time = 0
prev_pose = None
cmd_vel_x = None
cmd_vel_th = None

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

seq_no = 0

for row in cmd_vel_reader:
    cmd_vel_time = int(row['%time'])

    pose = prev_pose
    # There are many GT pose readings for each cmd_vel, so let's not
    # worry about interpolating to try to synchronize these timestamps.
    for pose_row in pose_reader:
        pose_time = int(pose_row['field.header.stamp'])

        # Do this at every timestep so that if we hit the end of the file
        # we can still collect the last example
        x = float(pose_row['field.pose.position.x'])
        y = float(pose_row['field.pose.position.y'])
        z = float(pose_row['field.pose.position.z'])
        qx = float(pose_row['field.pose.orientation.x'])
        qy = float(pose_row['field.pose.orientation.y'])
        qz = float(pose_row['field.pose.orientation.z'])
        qw = float(pose_row['field.pose.orientation.w'])
        pose = np.array([x,y,z,qx,qy,qz,qw])

        if pose_time > cmd_vel_time:
            break

    # sanity check this was 100ms
    time_diff = cmd_vel_time - prev_cmd_vel_time
    if time_diff == 100 * 1000 * 1000:
        pass
        # Earlier, I stored deltas between subsequent poses. For multistep evaluation this does not really make sense.
        # guaranteed not to execute for the first cmd_vel
        #data_row = np.concatenate([[cmd_vel_x, cmd_vel_th], pose_difference(pose, prev_pose)])
        #data.append(data_row)
    elif time_diff != 0:
        print("WARNING: Weird time diff", time_diff//1000000)
        seq_no += 1

    cmd_vel_x = float(row['field.linear.x'])
    cmd_vel_th = float(row['field.angular.z'])
    data_row = np.concatenate([[seq_no, cmd_vel_x, cmd_vel_th], xyt_pose(pose)])
    data.append(data_row)

    prev_cmd_vel_time = cmd_vel_time
    prev_pose = pose

np.savetxt(target + '/np.txt', data)
cmd_vel_file.close()
pose_file.close()
