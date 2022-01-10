#!/usr/bin/env python3

# Don't run training if unit tests fail

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

if __name__ == "__main__":
    from models import *
    from models.torch_models.models import train as torch_train
    from models.torch_models.models import evaluate as torch_evaluate
    from data_utils import load_dataset, SequenceDataset
else:
    from .models import *
    from .models.torch_models.models import train as torch_train
    from .models.torch_models.models import evaluate as torch_evaluate
    from .data_utils import load_dataset, SequenceDataset

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You need to specify DATASET_NAME")
        sys.exit(0)
    target = sys.argv[1]
    viz_seq_no = -1
    viz_start_idx = 50
    if len(sys.argv) == 4:
        viz_seq_no = int(sys.argv[2])
        viz_start_idx = int(sys.argv[3])


    #np.set_printoptions(precision=3, suppress=True)
    seqs, time_seqs = load_dataset(target)

    # Together, the total dimension of each training row is D+H+P
    # D: the dimension of the control input
    # H: the dimension of the odom information
    # P: the dimension of the ground truth pose / twist information
    # Different models in models.py can decide for themselves which of these are
    # input and output (independent and dependent) variables.
    is_ackermann = False
    if target == "sim_data":
        D = 2 # cmd_vel in the form dx, dtheta
        H = 0
        # P = 3 for poses (x, y, theta)
        P = 3
    elif target == "sim_odom_twist":
        D = 2
        # This also includes odom measurements of dx, dtheta
        H = 2
        # This also includes dx, dy, dtheta
        P = 6
    elif target.startswith("rzr_"):
        is_ackermann = True
        D = 3 # throttle, brake, steer (multiplied by -1 if we're in reverse)
        H = 2
        P = 6
    else:
        print("Unknown target dataset")
        sys.exit(0)

    N_SEQS = len(seqs)
    print("Found seqs:", N_SEQS)
    print("Of lengths:", list(map(lambda s: len(s), seqs)))
    train_N_SEQS = N_SEQS * 4 // 5
    test_N_SEQS = N_SEQS - train_N_SEQS
    assert(test_N_SEQS > 0)

    assert seqs[0].shape[1] == D+H+P, \
        f"Expected shape of each element of each sequence to be {D+H+P} but got {seqs[0].shape[1]}"
    train_set = seqs[:train_N_SEQS]
    test_set = seqs[train_N_SEQS:]
    train_time_set = None
    test_time_set = None
    if time_seqs is not None:
        train_time_set = time_seqs[:train_N_SEQS]
        test_time_set = time_seqs[train_N_SEQS:]

    print("first train seq shape:", train_set[0].shape)
    print("first test seq shape:", test_set[0].shape)

    N_TRAIN_STEPS = 1
    N_EVAL_STEPS = 20

    # Qualitative visualization
    viz_seq = seqs[viz_seq_no]
    n_steps = 20
    t_start = viz_start_idx-20 if viz_start_idx > 20 else 0
    t_end = viz_start_idx+n_steps+1

    if P in {6, 13}:
        gt_twist_model = GTTwistModel(D=D, H=H, P=P)
        gt_twist_score_pretraining = gt_twist_model.evaluate(test_set, n_steps=N_EVAL_STEPS)
        _, _, gmx, gmy = gt_twist_model.compare_qualitative(
                viz_seq, start_idx=viz_start_idx, n_steps=n_steps)
        line, = plt.plot(gmx, gmy, color='green', label='Rollout ground truth twist')
        gt_twist_model.train(train_set, n_steps=N_TRAIN_STEPS)
        gt_twist_score_posttraining = gt_twist_model.evaluate(test_set, n_steps=N_EVAL_STEPS)
        print("Ground truth twist baseline: untrained {} trained {}".format(
                gt_twist_score_pretraining, gt_twist_score_posttraining))
        print("Ground truth twist, weights post training")
        print(gt_twist_model.w)

    mean_model = MeanModel(D=D, H=H, P=P)
    mean_model.train(train_set, n_steps=N_TRAIN_STEPS)
    mean_score = mean_model.evaluate(test_set, n_steps=N_EVAL_STEPS)
    print("Mean model:    ", mean_score)

    # TODO do I make a new model type or add another param to these models?
    if D == 2:
        uni_model = UnicycleModel(D=D, H=H, P=P, delay_steps=1)
        uni_model.train(train_set, n_steps=N_TRAIN_STEPS)
        _, _, umx, umy = uni_model.compare_qualitative(
                viz_seq, start_idx=viz_start_idx, n_steps=n_steps)
        line, = plt.plot(umx, umy, color='red', label='Unicycle model')
        uni_score = uni_model.evaluate(test_set, n_steps=N_EVAL_STEPS)
        print("Unicycle model:", uni_score)

    linear_model = LinearModel(D=D, H=H, P=P, delay_steps=1, lambda_=1e-2)
    linear_model.train(train_set, n_steps=N_TRAIN_STEPS)
    tx, ty, mx, my = linear_model.compare_qualitative(
            viz_seq, start_idx=viz_start_idx, n_steps=n_steps)
    line, = plt.plot(tx[t_start:t_end], ty[t_start:t_end], color='black', label='Ground truth')
    line, = plt.plot(mx, my, color='blue', label='Fitted linear model')
    linear_score = linear_model.evaluate(test_set, n_steps=N_EVAL_STEPS)
    print("Linear model:  ", linear_score)
    print("Linear weights")
    print(linear_model.w)

    expected = np.array([[ 0.09578356, -0.00020478,  0.00039695],
                         [ 0.00237972,  0.00028133,  0.03389382]])
    if target == "sim_data" and not np.allclose(linear_model.w, expected):
        print("ERROR: Did not get expected weights for simple linear model")

    plt.legend()
    plt.axis('equal')
    plt.savefig('out.png')

    if not is_ackermann:
        plt.clf()
        plt.axis('equal')
        x,y = linear_model.concat_seqs(seqs,1)
        plt.xlabel("Commanded heading rate (rad/s)")
        plt.ylabel("Actual heading rate (rad/s)")
        # Note: I don't think the indices 1 and 2 are correct except for the `sim_data` dataset
        plt.scatter(x[:,1],y[:,2]*10)
        plt.savefig('z_vs_zcmd_out.png')

    # Train simple torch model
    x_features = torch.zeros((D + H + P), dtype=torch.bool)
    x_features[:D] = True
    y_features = torch.zeros((D + H + P), dtype=torch.bool)
    y_features[D + H:D + H + 3] = True
    train_loader = DataLoader(
        SequenceDataset(train_set, x_features, y_features, delay_steps=1, n_steps=N_TRAIN_STEPS, time_data=train_time_set)
    )
    val_loader = DataLoader(
        SequenceDataset(train_set, x_features, y_features, delay_steps=1, n_steps=N_EVAL_STEPS, time_data=test_time_set)
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(D, 16),
        torch.nn.Sigmoid(),
        torch.nn.Linear(16, 16),
        torch.nn.Sigmoid(),
        torch.nn.Linear(16, 3),
    )
    torch_train(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=torch.nn.MSELoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        device="cpu",
    )