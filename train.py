#!/home/nhatch2/miniconda3/bin/python3

# Don't run training if unit tests fail
import unit_test

import sys
import numpy as np
from models import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You need to specify DATASET_NAME")
        sys.exit(0)
    target = sys.argv[1]

    #np.set_printoptions(precision=3, suppress=True)

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

    if target == "sim_data":
        D = 2
        # P = 3 for poses (x, y, theta)
        P = 3
    elif target == "sim_odom_twist":
        # This also includes odom measurements of dx, dtheta
        D = 4
        # This also includes dx, dy, dtheta
        P = 6
    else:
        print("Unknown target dataset")
        sys.exit(0)

    N_SEQS = len(seqs)
    print("Found seqs:", N_SEQS)
    print("Of lengths:", list(map(lambda s: len(s), seqs)))
    print("With num_features:", D)
    train_N_SEQS = N_SEQS * 4 // 5
    test_N_SEQS = N_SEQS - train_N_SEQS
    assert(test_N_SEQS > 0)

    assert(seqs[0].shape[1] == D+P)
    x_train = list(map(lambda s: s[:,:D], seqs[:train_N_SEQS]))
    y_train = list(map(lambda s: s[:,D:], seqs[:train_N_SEQS]))
    x_test = list(map(lambda s: s[:,:D], seqs[train_N_SEQS:]))
    y_test = list(map(lambda s: s[:,D:], seqs[train_N_SEQS:]))

    print("first train seq shape:", x_train[0].shape)
    print("first test seq shape:", x_test[0].shape)

    N_TRAIN_STEPS = 1
    N_EVAL_STEPS = 20

    mean_model = MeanModel()
    mean_model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
    mean_score = mean_model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
    print("Mean model:    ", mean_score)

    uni_model = UnicycleModel(num_features=D, delay_steps=1)
    uni_model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
    uni_score = uni_model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
    print("Unicycle model:", uni_score)

    ''' This model doesn't make sense because it is using odom data from the future.
    linear_model = LinearModel(num_features=D, delay_steps=1)
    linear_model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
    linear_score = linear_model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
    print("Linear model:  ", linear_score)

    print("linear weights")
    print(linear_model.w)
    '''

    linear_no_odom_model = LinearModel(num_features=D, delay_steps=1, ignore_indices=[2,3])
    linear_no_odom_model.train(x_train, y_train, n_steps=N_TRAIN_STEPS)
    linear_no_odom_score = linear_no_odom_model.evaluate(x_test, y_test, n_steps=N_EVAL_STEPS)
    print("Linear model (no odom):  ", linear_no_odom_score)
    print("linear weights (no odom)")
    print(linear_no_odom_model.w)

    expected = np.array([[ 0.09578356, -0.00020478,  0.00039695],
                         [ 0.00237972,  0.00028133,  0.03389382]])
    if target == "sim_data" and not np.allclose(linear_no_odom_model.w, expected):
        print("ERROR: Did not get expected weights for simple linear model")

    test_seq_no = 0
    start_idx = 100
    n_steps = 20
    t_start = start_idx-20 if start_idx > 20 else 0
    t_end = start_idx+n_steps+1
    tx, ty, mx, my = linear_no_odom_model.compare_qualitative(
            x_test[test_seq_no], y_test[test_seq_no], start_idx=start_idx, n_steps=n_steps)
    _, _, umx, umy = uni_model.compare_qualitative(
            x_test[test_seq_no], y_test[test_seq_no], start_idx=start_idx, n_steps=n_steps)
    plt.plot(tx[t_start:t_end], ty[t_start:t_end], color='black')
    plt.plot(mx, my, color='blue')
    plt.plot(umx, umy, color='red')
    plt.savefig('out.png')
