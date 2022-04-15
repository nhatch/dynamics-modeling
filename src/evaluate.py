import argparse
import numpy as np
import torch

from rosbag2torch.bag_processing.sequence_readers import ASyncSequenceReader

from rosbag2torch.bag_processing.filters import ForwardFilter, PIDInfoFilter
from rosbag2torch import LookaheadSequenceDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def unroll_y(y: np.ndarray, t: np.ndarray):
    thetas = np.cumsum(y[:, 1] * t)

    along_vecs = []

    for theta, vx, dt in zip(thetas, y[:, 0], t):
        along_vec = np.array([np.cos(theta), np.sin(theta)])
        along_vec /= np.linalg.norm(along_vec)
        along_vec *= vx * dt
        along_vecs.append(along_vec)

    return np.cumsum(np.array(along_vecs), axis=0)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('bag_path', nargs=1, type=str, help="Path to bag file to evaluate model on.")
    parser.add_argument('model', nargs=1, type=str, help="Path to model file to evaluate. Should be a torch.jit.scripted .pt file.")

    args = parser.parse_args()

    model = torch.jit.load(args.model[0])
    model.eval()

    # x_features = ["state", "control"]
    x_features = ["control", "state"]
    y_features = ["target"]

    filters = [ForwardFilter(), PIDInfoFilter()]

    reader = ASyncSequenceReader(x_features + y_features, features_to_record_on=y_features, filters=filters)

    reader.extract_bag_data(args.bag_path[0])

    # Eval sequence is the longest from among sequences
    eval_seq = None
    eval_seq_len = 0
    for seq in reader.sequences:
        key = list(seq.keys())[0]
        if len(seq[key]) > eval_seq_len:
            eval_seq = seq
            eval_seq_len = len(seq[key])

    dataset = LookaheadSequenceDataset([eval_seq], x_features, y_features, delay_steps=1)

    data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

    y_pred_all = []
    y_true_all = []
    ts_all = []

    for batch in data_loader:
        x, y, dt = batch
        y_pred = model(x) * dt + x[:, -2:]

        y_pred_all.extend(y_pred.detach().numpy())
        y_true_all.extend(y.detach().numpy())
        ts_all.extend(dt.detach().numpy())

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)

    print(f"Average MSE: {np.mean(np.square(y_pred_all - y_true_all))}")
    ts_all = np.array(ts_all)

    fig, axs = plt.subplots(2, 2, sharey="col")

    # Smoothen the data
    smooth_dx = True
    if smooth_dx:
        from scipy.signal import savgol_filter
        y_pred_all[:, 0] = savgol_filter(y_pred_all[:, 0], window_length=31, polyorder=3, axis=0, mode="nearest")

    for ax_row, y, title in zip(axs, [y_pred_all, y_true_all], ["pred", "true"]):
        ax_row[0].plot(np.cumsum(ts_all), y[:, 0])
        ax_row[0].set_title(f"{title} - dx")

        ax_row[1].plot(np.cumsum(ts_all), y[:, 1])
        ax_row[1].set_title(f"{title} - dTheta")

    plt.show()

    y_pred_all = unroll_y(y_pred_all, ts_all)
    y_true_all = unroll_y(y_true_all, ts_all)

    plt.plot(y_pred_all[:, 0], y_pred_all[:, 1], label="Pred")
    plt.plot(y_true_all[:, 0], y_true_all[:, 1], label="True")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
