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
D = 2
train_N_SEQS = N_SEQS // 5 * 4
test_N_SEQS = N_SEQS - train_N_SEQS
assert(test_N_SEQS > 0)

all_train = np.concatenate(seqs[:train_N_SEQS], 0)
x_train = all_train[:,:D]
y_train = all_train[:,D:]
all_test = np.concatenate(seqs[train_N_SEQS:], 0)
x_test =  all_test[:,:D]
y_test =  all_test[:,D:]

print("train shape:", x_train.shape)
print("test shape:", x_test.shape)

# TODO put more weight on angular error? especially z
WEIGHTS = np.array([10,10,10,100,100,100])
#WEIGHTS = np.array([10,0.0001,0.0001,0.0001,0.0001,0.0001])

class MeanModel:
    def __init__(self):
        self.mean = None

    def train(self, x, y):
        assert(x.shape[0] == y.shape[0])
        self.mean = y.mean(axis=0)
        assert(self.mean.shape == (y.shape[1],))

    def predict(self, x):
        return np.tile(self.mean, (x.shape[0], 1))

# I think we don't want an affine model, since for zero input
# we should actually get zero output.
class LinearModel:
    def __init__(self):
        self.w = None

    def train(self, x, y):
        weighted_y = y * WEIGHTS
        assert(weighted_y.shape == y.shape)
        #weighted_w = np.linalg.inv(x.T @ x) @ (x.T @ weighted_y)
        weighted_w = np.linalg.solve(x.T @ x, x.T @ weighted_y)

        assert(weighted_w.shape == (x.shape[1], weighted_y.shape[1]))
        self.w = weighted_w / WEIGHTS
        assert(self.w.shape == weighted_w.shape)

    def predict(self, x):
        return x @ self.w

def evaluate(model, x, y):
    pred = model.predict(x)
    assert(pred.shape == y.shape)
    weighted_diff = (y - pred) * WEIGHTS
    assert(weighted_diff.shape == pred.shape)
    errs = 0.5 * (weighted_diff * weighted_diff).sum(axis=1)
    assert(errs.shape == (x.shape[0],))
    return errs.mean()

model = MeanModel()
model.train(x_train, y_train)
score = evaluate(model, x_test, y_test)
print(score)

model = LinearModel()
model.w = np.array([[0.1, 0, 0, 0, 0, 0], [0, 0, 0, 0.1, 0, 0]])
score = evaluate(model, x_test, y_test)
print(score)

model.train(x_train, y_train)
score = evaluate(model, x_test, y_test)
print(score)

print("linear weights")
print(model.w)
