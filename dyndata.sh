#!/bin/bash
rosbag record -e '/tf|.*cmd_vel.*|.*input.*|.*vehicle|.*ground_truth.*|.*odom|.*odometry/filtered'
