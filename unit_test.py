#!/home/nhatch2/miniconda3/bin/python3

import numpy as np
from train import LinearModel

model = LinearModel()

queries = np.array([
    [1,1, np.pi/4],
    [0,3, np.pi/2]])
references = np.array([
    [0.5, -1, np.pi*1.5],
    [0, 0, np.pi]])
relative = model.relative_pose(queries, references)
expected_relative = np.array([
    [-2.0, 0.5, np.pi * 3/4],
    [0, -3, -np.pi/2]])
assert(np.allclose(relative, expected_relative))

# Poses in world frame
# d_x, d_th, p_x, p_y, p_th
rotate_seq = np.array([
    [0.0, -1.0, 0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0, 0.0, 0.5],
    [0.0, -1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0, 2*np.pi - 0.5],
    [0.0, -1.0, 0.0, 0.0, 2*np.pi - 1.0]])

rotate_nstep_1 = model.get_n_step_targets(rotate_seq[:,2:], 1)
rotate_nstep_2 = model.get_n_step_targets(rotate_seq[:,2:], 2)

rotate_true_1 = np.array([
    [0.0, 0.0, -0.5],
    [0.0, 0.0, -0.5],
    [0.0, 0.0, -0.5],
    [0.0, 0.0, -0.5]])

rotate_true_2 = np.array([
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0]])

assert(np.allclose(rotate_nstep_1, rotate_true_1))
assert(np.allclose(rotate_nstep_2, rotate_true_2))

circle_seq = np.array([
    [1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.2, 0.1, 0.3],
    [1.0, 1.0, 0.4, 0.2, 0.6],
    [1.0, 1.0, 0.5, 0.3, 0.9],
    [1.0, 1.0, 0.6, 0.5, 1.2]])

circle_nstep_1 = model.get_n_step_targets(circle_seq[:,2:], 1)
circle_nstep_2 = model.get_n_step_targets(circle_seq[:,2:], 2)

circle_true_1 = np.array([
    [0.2, 0.1, 0.3],
    model.relative_pose(np.array([[0.4, 0.2, 0.6]]), np.array([[0.2, 0.1, 0.3]])).flatten(),
    model.relative_pose(np.array([[0.5, 0.3, 0.9]]), np.array([[0.4, 0.2, 0.6]])).flatten(),
    model.relative_pose(np.array([[0.6, 0.5, 1.2]]), np.array([[0.5, 0.3, 0.9]])).flatten()])

circle_true_2 = np.array([
    [0.4, 0.2, 0.6],
    model.relative_pose(np.array([[0.5, 0.3, 0.9]]), np.array([[0.2, 0.1, 0.3]])).flatten(),
    model.relative_pose(np.array([[0.6, 0.5, 1.2]]), np.array([[0.4, 0.2, 0.6]])).flatten()])

assert(np.allclose(circle_nstep_1, circle_true_1))
assert(np.allclose(circle_nstep_2, circle_true_2))

circle_rollout = model.rollout_one_steps(circle_nstep_1, 4)
print(circle_rollout)
assert(np.allclose(circle_rollout, circle_seq[-1, 2:]))

