#!/home/nhatch2/miniconda3/bin/python3

import numpy as np
from models import *

if __name__ == "__main__":
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
