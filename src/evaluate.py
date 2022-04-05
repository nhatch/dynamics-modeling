import numpy as np
import torch
from data_utils.bag_processing.sequence_readers import ASyncSequenceReader
from data_utils.bag_processing.filters import ForwardFilter, PIDInfoFilter
from data_utils.datasets.torch_lookahead import LookaheadSequenceDataset
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
    model = torch.jit.load('model_scripted.pt')
    model.eval()

    x_features = ["state", "control"]
    y_features = ["target"]

    filters = [ForwardFilter(), PIDInfoFilter()]

    reader = ASyncSequenceReader(x_features + y_features, features_to_record_on=y_features, filters=filters)

    reader.extract_bag_data("datasets/rzr_real/auton_medium_2022-02-24-21-28-04.bag")

    # Eval sequence is the longest from among sequences
    eval_seq = None
    eval_seq_len = 0
    for seq in reader.sequences:
        key = list(seq.keys())[0]
        if len(seq[key]) > eval_seq_len:
            eval_seq = seq
            eval_seq_len = len(seq[key])

    dataset = LookaheadSequenceDataset([eval_seq], x_features, y_features, delay_steps=0, n_steps=1)

    data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

    y_pred_all = []
    y_true_all = []
    ts_all = []

    for batch in data_loader:
        x, y, t = batch
        y_pred = model(x) * t

        y_pred_all.extend(y_pred.detach().numpy())
        y_true_all.extend(y.detach().numpy())
        ts_all.extend(t.detach().numpy())

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    ts_all = np.array(ts_all)

    fig, axs = plt.subplots(2, 2)
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
