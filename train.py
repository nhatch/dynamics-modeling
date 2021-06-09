#!/home/nhatch2/miniconda3/bin/python3

# Don't run training if unit tests fail
import unit_test

import numpy as np
from models import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #np.set_printoptions(precision=3, suppress=True)

    target = 'sim_data'
    X = np.loadtxt('datasets/' + target + '/np.txt')
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

    mean_model = MeanModel()
    mean_model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
    mean_score = mean_model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
    print("Mean model:    ", mean_score)

    uni_model = UnicycleModel(delay_steps=1)
    uni_model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
    uni_score = uni_model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
    print("Unicycle model:", uni_score)

    linear_model = LinearModel(delay_steps=1)
    linear_model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
    linear_score = linear_model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
    print("Linear model:  ", linear_score)

    print("linear weights")
    print(linear_model.w)

    test_seq_no = 0
    start_idx = 100
    n_steps = 20
    t_start = start_idx-20 if start_idx > 20 else 0
    t_end = start_idx+n_steps+1
    tx, ty, mx, my = linear_model.compare_qualitative(
            x_test[test_seq_no], y_test[test_seq_no], start_idx=start_idx, n_steps=n_steps)
    _, _, umx, umy = uni_model.compare_qualitative(
            x_test[test_seq_no], y_test[test_seq_no], start_idx=start_idx, n_steps=n_steps)
    plt.plot(tx[t_start:t_end], ty[t_start:t_end], color='black')
    plt.plot(mx, my, color='blue')
    plt.plot(umx, umy, color='red')
    plt.savefig('out.png')
