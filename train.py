#!/home/nhatch2/miniconda3/bin/python3

import numpy as np
np.set_printoptions(precision=3, suppress=True)

target = 'sim_data'
X = np.loadtxt(target + '/np.txt')
N = X.shape[0]
seqs = []
seq = []
seq_no = 0
for row in X:
    data = row[1:].reshape((1,-1))
    if row[0] != seq_no:
        if len(seq) > 1:
            seqs.append(seq)
        seq = data
        seq_no = row[0]
    else:
        seq = np.concatenate([seq, data], 0)

N_SEQS = len(seqs)
print("Found seqs:", N_SEQS)
print("Of lengths:", list(map(lambda s: len(s), seqs)))
D = 2
train_N_SEQS = N_SEQS // 5 * 4
test_N_SEQS = N_SEQS - train_N_SEQS
assert(test_N_SEQS > 0)

assert(seqs[0].shape[1] == 5)
x_train = list(map(lambda s: s[:,:2], seqs[:train_N_SEQS]))
y_train = list(map(lambda s: s[:,2:], seqs[:train_N_SEQS]))
x_test = list(map(lambda s: s[:,:2], seqs[train_N_SEQS:]))
y_test = list(map(lambda s: s[:,2:], seqs[train_N_SEQS:]))

print("first train seq shape:", x_train[0].shape)
print("first test seq shape:", x_test[0].shape)

# We used to try to predict full 6D pose
#WEIGHTS = np.array([10,10,10,100,100,100])
#WEIGHTS = np.array([10,0.0001,0.0001,0.0001,0.0001,0.0001])
# This is just x,y,theta
WEIGHTS = np.array([10,10,100])

# I think we don't want an affine model, since for zero input
# we should actually get zero output.
class LinearModel:
    def __init__(self):
        self.w = None
        self.train_n_steps = None

    def rollout_y_seq(self, yy, n_steps):
        assert(yy.shape[1] == 3)
        seq_len = len(yy) - n_steps
        target_deltas = np.zeros_like(yy[:seq_len,:])
        # Rather than just taking the difference between t and t+n_steps,
        # we integrate over those n steps. This is just to make sure we
        # get the correct angle difference (i.e. not off by k*2*pi).
        for t in range(n_steps):
            delta = yy[t+1:t+1+seq_len,:] - yy[t:t+seq_len,:]
            world_delta = np.zeros_like(delta)
            curr_angles = target_deltas[:,2]
            world_delta[:,0] = np.cos(curr_angles) * delta[:,0] - np.sin(curr_angles) * delta[:,1]
            world_delta[:,1] = np.sin(curr_angles) * delta[:,0] + np.cos(curr_angles) * delta[:,1]
            # Column 2 is the angle
            # Within one timestep we should use the smallest representation of the angle
            world_delta[:,2] = np.arcsin(np.sin(delta[:,2]))
            target_deltas += world_delta
        return target_deltas

    def concat_seqs(self, x_seqs, y_seqs, n_steps):
        n_seqs = len(x_seqs)
        assert(n_seqs == len(y_seqs))
        rollout_x_seqs = []
        rollout_y_seqs = []
        for s in range(n_seqs):
            xx = x_seqs[s]
            yy = y_seqs[s]
            seq_len = len(yy) - n_steps
            assert(len(xx) == len(yy))
            assert(xx.shape[1] == 2)
            target_deltas = self.rollout_y_seq(yy, n_steps)
            # TODO if n_steps > 1, we should concatenate multiple commands together
            rollout_x_seqs.append(xx[:seq_len,:])
            rollout_y_seqs.append(target_deltas)

        train_x = np.concatenate(rollout_x_seqs, 0)
        train_y = np.concatenate(rollout_y_seqs, 0)
        return train_x, train_y

    def train(self, x_seqs, y_seqs, n_steps=1):
        self.train_n_steps = n_steps
        if n_steps != 1:
            print("Warning: Linear model training with n_steps > 1 is not supported!")
        train_x, train_y = self.concat_seqs(x_seqs, y_seqs, n_steps)
        weighted_y = train_y * WEIGHTS
        assert(weighted_y.shape == train_y.shape)
        assert(len(train_x) == len(train_y))
        weighted_w = np.linalg.solve(train_x.T @ train_x, train_x.T @ weighted_y)

        assert(weighted_w.shape == (train_x.shape[1], weighted_y.shape[1]))
        self.w = weighted_w / WEIGHTS
        assert(self.w.shape == weighted_w.shape)

    def predict(self, x_seqs, n_steps=1):
        if self.train_n_steps != 1:
            print("Warning: Trained with n_steps != 1; don't know how to predict")
        pred_seqs = []
        for xx in x_seqs:
            pred_seqs.append(self.predict_seq(xx, n_steps))
        return pred_seqs

    def predict_one_steps(self, xx):
        return xx @ self.w

    def predict_seq(self, xx, n_steps):
        seq_len = len(xx) - n_steps
        one_steps = self.predict_one_steps(xx[:-1,:])
        preds = np.zeros((seq_len, one_steps.shape[1]))
        for t in range(n_steps):
            curr_angles = preds[:,2]
            summand = one_steps[t:t+seq_len,:]
            assert(summand.shape == (seq_len, 3))
            world_summand = np.zeros_like(summand)
            world_summand[:,0] = np.cos(curr_angles) * summand[:,0] - np.sin(curr_angles) * summand[:,1]
            world_summand[:,1] = np.sin(curr_angles) * summand[:,0] + np.cos(curr_angles) * summand[:,1]
            world_summand[:,2] = summand[:,2]
            preds += world_summand
        return preds

    def evaluate(self, x_seqs, y_seqs, n_steps=1):
        n_seqs = len(x_seqs)
        assert(n_seqs == len(y_seqs))
        pred_seqs = self.predict(x_seqs, n_steps)
        assert(len(pred_seqs) == len(y_seqs))
        all_errs = []
        for s in range(n_seqs):
            pp = pred_seqs[s]
            yy = y_seqs[s]
            rollout = self.rollout_y_seq(yy, n_steps)
            weighted_diff = (rollout - pp) * WEIGHTS
            assert(weighted_diff.shape == pp.shape)
            errs = 0.5 * (weighted_diff * weighted_diff).sum(axis=1)
            assert(errs.shape == (yy.shape[0] - n_steps,))
            all_errs.append(errs)
        concat_errs = np.concatenate(all_errs)
        return concat_errs.mean()

class MeanModel(LinearModel):
    def __init__(self):
        self.mean = None
        self.train_n_steps = 1

    def train(self, x_seqs, y_seqs, n_steps=1):
        train_x, train_y = self.concat_seqs(x_seqs, y_seqs, n_steps)
        self.mean = train_y.mean(axis=0)
        assert(self.mean.shape == (train_y.shape[1],))

    def predict_one_steps(self, xx):
        return np.tile(self.mean, (xx.shape[0], 1))

class UnicycleModel(LinearModel):
    def train(self, x_seqs, y_seqs, n_steps=1):
        self.train_n_steps = 1
        self.w = np.array([[0.1, 0, 0], [0, 0, 0.1]])


N_TRAIN_STEPS = 1
N_EVAL_STEPS = 20

model = MeanModel()
model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
score = model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
print("Mean model:    ", score)

model = UnicycleModel()
model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
score = model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
print("Unicycle model:", score)

model = LinearModel()
model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
score = model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
print("Linear model:  ", score)

print("linear weights")
print(model.w)
