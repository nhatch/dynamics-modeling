#!/bin/bash
root=datasets
bagfile=$1
target=$2
mkdir ${root}/${target}
rostopic echo -b $bagfile -p /rzr/input > ${root}/${target}/input.txt
rostopic echo -b $bagfile -p /rzr/odom > ${root}/${target}/odom.txt
rostopic echo -b $bagfile -p /unity_command/ground_truth/rzr/pose > ${root}/${target}/gt_pose.txt
rostopic echo -b $bagfile -p /unity_command/ground_truth/rzr/twist > ${root}/${target}/gt_twist.txt
