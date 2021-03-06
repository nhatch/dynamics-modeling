#!/bin/bash
root=datasets
name=rzr
bagfile=$1
target=$2
mkdir ${root}/${target}
rostopic echo -b $bagfile -p /${name}/input > ${root}/${target}/input.txt
rostopic echo -b $bagfile -p /${name}/odom > ${root}/${target}/odom.txt
rostopic echo -b $bagfile -p /unity_command/ground_truth/${name}/pose > ${root}/${target}/gt_pose.txt
rostopic echo -b $bagfile -p /unity_command/ground_truth/${name}/twist > ${root}/${target}/gt_twist.txt
