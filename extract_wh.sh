#!/bin/bash
root=datasets
name=warthog
bagfile=$1
target=$2
mkdir ${root}/${target}
rostopic echo -b $bagfile -p /${name}/cmd_vel > ${root}/${target}/cmd_vel.txt
rostopic echo -b $bagfile -p /${name}/warthog_velocity_controller/odom > ${root}/${target}/odom.txt
rostopic echo -b $bagfile -p /unity_command/ground_truth/${name}/pose > ${root}/${target}/gt_pose.txt
rostopic echo -b $bagfile -p /unity_command/ground_truth/${name}/twist > ${root}/${target}/gt_twist.txt
