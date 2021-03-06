#!/usr/bin/env python3

import numpy as np

# We used to try to predict full 6D pose
#WEIGHTS = np.array([10,10,10,100,100,100])
#WEIGHTS = np.array([10,0.0001,0.0001,0.0001,0.0001,0.0001])
# This is just x,y,theta
WEIGHTS = np.array([10,10,100])

# I think we don't want an affine model, since for zero input
# we should actually get zero output.
class LinearModel:
    def __init__(self, D, H, P, delay_steps):
        self.w = None
        self.train_n_steps = None
        self.delay_steps = delay_steps
        self.features_to_use = np.zeros(D+H+P, dtype=np.bool)
        self.features_to_predict = np.zeros(D+H+P, dtype=np.bool)
        self.features_to_use[:D] = True
        self.features_to_predict[D+H:D+H+3] = True

    def relative_pose(self, query_pose, reference_pose):
        diff = query_pose - reference_pose
        distance = np.linalg.norm(diff[:,:2], axis=1)
        direction = np.arctan2(diff[:,1], diff[:,0])
        relative_direction = direction - reference_pose[:,2]
        angle_diff = diff[:,2]
        minimized_angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        return np.array([distance*np.cos(relative_direction),
                         distance*np.sin(relative_direction),
                         minimized_angle_diff]).T

    def get_n_step_targets(self, yy, n_steps):
        assert(yy.shape[1] == 3 or yy.shape[1] == 6)
        targets = self.relative_pose(yy[n_steps:,:3], yy[:-n_steps,:3])
        # We can't do predictions for the first few steps if those depend
        # on data from previous timesteps.
        targets = targets[self.delay_steps:,:]
        return targets

    def concat_seqs(self, seqs, n_steps):
        n_seqs = len(seqs)
        rollout_x_seqs = []
        rollout_y_seqs = []
        for s in range(n_seqs):
            xx = seqs[s][:,self.features_to_use]
            yy = seqs[s][:,self.features_to_predict]
            seq_len = len(yy) - n_steps - self.delay_steps
            assert(len(xx) == len(yy))
            assert(xx.shape[1] == self.features_to_use.sum())
            target_deltas = self.get_n_step_targets(yy, n_steps)
            # TODO if n_steps > 1, we should concatenate multiple commands together
            rollout_x_seqs.append(xx[:seq_len,:])
            rollout_y_seqs.append(target_deltas)

        train_x = np.concatenate(rollout_x_seqs, 0)
        train_y = np.concatenate(rollout_y_seqs, 0)
        return train_x, train_y

    def train(self, seqs, n_steps=1):
        self.train_n_steps = n_steps
        if n_steps != 1:
            print("Warning: Linear model training with n_steps > 1 is not supported!")
        train_x, train_y = self.concat_seqs(seqs, n_steps)
        weighted_y = train_y * WEIGHTS
        assert(weighted_y.shape == train_y.shape)
        assert(len(train_x) == len(train_y))
        weighted_w = np.linalg.solve(train_x.T @ train_x, train_x.T @ weighted_y)

        assert(weighted_w.shape == (train_x.shape[1], weighted_y.shape[1]))
        self.w = weighted_w / WEIGHTS
        assert(self.w.shape == weighted_w.shape)

    def predict_one_steps(self, xx):
        return xx @ self.w

    def rollout_one_steps(self, one_steps, n_steps):
        # TODO dealing with delays will be more complicated with dynamics models that
        # depend on the current state
        seq_len = len(one_steps) - n_steps + 1 - self.delay_steps
        preds = np.zeros((seq_len, one_steps.shape[1]))
        for t in range(n_steps):
            curr_angles = preds[:,2]
            summand = one_steps[t:t+seq_len,:]
            # TODO state for Ackermann vehicles should be 6D
            assert(summand.shape == (seq_len, 3))
            world_summand = np.zeros_like(summand)
            world_summand[:,0] = np.cos(curr_angles) * summand[:,0] - np.sin(curr_angles) * summand[:,1]
            world_summand[:,1] = np.sin(curr_angles) * summand[:,0] + np.cos(curr_angles) * summand[:,1]
            world_summand[:,2] = summand[:,2]
            preds += world_summand
        return preds

    # Relative to rollout_one_steps, this method:
    # - Saves the intermediate steps of the sequence
    # - Only rolls out one sequence (for the entire length of one_steps, rather than seq_len seqs each of length n_steps)
    # - Starts from a given start state rather than 0,0,0
    # - Doesn't deal with delay_steps
    def rollout_single_sequence(self, one_steps, start_state):
        n_steps = one_steps.shape[0]
        seq = np.zeros((n_steps+1, one_steps.shape[1]))
        seq[0] = start_state
        for t in range(n_steps):
            curr_angle = seq[t,2]
            summand = one_steps[t,:]
            assert(summand.shape == (3,))
            world_summand = np.zeros_like(summand)
            world_summand[0] = np.cos(curr_angle) * summand[0] - np.sin(curr_angle) * summand[1]
            world_summand[1] = np.sin(curr_angle) * summand[0] + np.cos(curr_angle) * summand[1]
            world_summand[2] = summand[2]
            seq[t+1,:] = seq[t,:] + world_summand
        return seq

    def compare_qualitative(self, seq, start_idx, n_steps):
        xx = seq[:,self.features_to_use]
        yy = seq[:,self.features_to_predict]
        one_steps = self.predict_one_steps(xx[:-1,:])
        start_state = yy[start_idx,:3]
        ossi = start_idx - self.delay_steps
        relevant_one_steps = one_steps[ossi:ossi+n_steps]
        seq = self.rollout_single_sequence(relevant_one_steps, start_state)
        tx = yy[:,0]
        ty = yy[:,1]
        mx = seq[:,0]
        my = seq[:,1]
        return tx, ty, mx, my

    def predict_seq(self, xx, n_steps):
        one_steps = self.predict_one_steps(xx[:-1,:])
        return self.rollout_one_steps(one_steps, n_steps)

    def evaluate(self, seqs, n_steps=1):
        if self.train_n_steps != 1:
            print("Warning: Trained with n_steps != 1; don't know how to predict")
        n_seqs = len(seqs)
        all_errs = []
        for s in range(n_seqs):
            xx = seqs[s][:,self.features_to_use]
            yy = seqs[s][:,self.features_to_predict]
            pp = self.predict_seq(xx, n_steps)
            targets = self.get_n_step_targets(yy, n_steps)
            assert(targets.shape == pp.shape)
            #print(pp[:10,:])
            #print()
            #print(targets[:10,:])
            #print()
            weighted_diff = (targets - pp) * WEIGHTS / n_steps
            assert(weighted_diff.shape == pp.shape)
            errs = 0.5 * (weighted_diff * weighted_diff).sum(axis=1)
            assert(errs.shape == (yy.shape[0] - n_steps - self.delay_steps,))
            all_errs.append(errs)
        concat_errs = np.concatenate(all_errs)
        return concat_errs.mean()

class MeanModel(LinearModel):
    def __init__(self, D, H, P):
        super().__init__(D, H, P, 0)
        self.mean = None
        self.train_n_steps = 1

    def train(self, seqs, n_steps=1):
        train_x, train_y = self.concat_seqs(seqs, n_steps)
        self.mean = train_y.mean(axis=0)
        assert(self.mean.shape == (train_y.shape[1],))

    def predict_one_steps(self, xx):
        return np.tile(self.mean, (xx.shape[0], 1))

class UnicycleModel(LinearModel):
    def __init__(self, D, H, P, delay_steps):
        # Unicycle model doesn't account for velocity state (features 2 and 3)
        super().__init__(D, H, P, delay_steps)
        self.train_n_steps = 1

    def train(self, seqs, n_steps=1):
        dt = 0.1
        effective_wheel_base = 1.80
        measured_wheel_base = 1.08
        theta_factor = measured_wheel_base / effective_wheel_base
        self.w = dt * np.array([
            [1, 0, 0],
            [0, 0, theta_factor]
            ])

class GTTwistModel(LinearModel):
    def __init__(self, D, H, P):
        super().__init__(D, H, P, 0)
        assert(P == 6)
        self.features_to_use = np.zeros(D+H+P, dtype=np.bool)
        self.features_to_use[-3:] = True
        self.train_n_steps = 1

        # Before training, set a default
        dt = 0.1
        self.w = dt * np.eye(3)
