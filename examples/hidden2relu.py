import os
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from rosbag2torch import LookaheadDataset, filters, load_bags, readers
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from example_utils import reconstruct_poses_from_acc, reconstruct_poses_from_odoms

def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    epochs: int,
    val_loader: Optional[DataLoader] = None,
    verbose: bool = True,
    writer: Optional[SummaryWriter] = None,
):
    trange_epochs = trange(epochs, desc="Epochs", disable=not verbose, leave=True)
    for epoch in trange_epochs:
        running_loss = 0.0
        running_baseline_loss = 0.0
        for x, y, dt in tqdm(train_loader, disable=not verbose, desc="Train", leave=False):
            optimizer.zero_grad()
            # acceleration * dt + prev state
            # m / s^2 * s + m / s
            y_pred = model(x) * dt

            loss = criterion(y_pred, y - x[:, -2:])
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            running_baseline_loss += criterion(y, x[:, -2:]).detach().cpu().item()

        train_loss = running_loss / len(train_loader)
        train_baseline_loss = running_baseline_loss / len(train_loader)
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/train baseline (zero acc.)", train_baseline_loss, epoch)
        desc = f"Epochs Train Loss {train_loss:.4g} (0 acc.) Loss {train_baseline_loss:.4g}"

        if val_loader is not None:
            with torch.no_grad():
                val_running_loss = 0.0
                val_running_baseline_loss = 0.0
                for x, y, dt in tqdm(val_loader, disable=not verbose, desc="Validation", leave=False):
                    y_pred = model(x) * dt

                    loss = criterion(y_pred, y - x[:, -2:])

                    val_running_loss += loss.detach().cpu().item()
                    val_running_baseline_loss += criterion(y, x[:, -2:]).detach().cpu().item()


            val_loss = val_running_loss / len(val_loader)
            val_baseline_loss = val_running_baseline_loss / len(val_loader)
            if writer is not None:
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Loss/val baseline (zero acc.)", val_baseline_loss, epoch)
            desc += f" Val Loss {val_loss:.4g} (0 acc.) Loss {val_baseline_loss:.4g}"

        trange_epochs.set_description(desc)


def main():
    DELAY_STEPS = 15
    EPOCHS = 50
    TRAIN = False
    PLOT_VAL = True
    PLOT_LEN_ROLLOUT = 10  # seconds

    # # Async Reader
    # reader = readers.ASyncSequenceReader(
    #     ["control", "state", "target"],
    #     features_to_record_on=["control"],
    #     filters=[
    #         filters.ForwardFilter(),
    #         filters.PIDInfoFilter()
    #     ]
    # )

    # Fixed Timestamp Reader
    reader = readers.FixedIntervalReader(
        ["control", "state", "target"],
        log_interval=1 / 30.,
        filters=[
            filters.ForwardFilter(),
            filters.PIDInfoFilter()
        ]
    )

    val_sequences = load_bags("datasets/rzr_real_val", reader)

    model_name = f"models/example_hidden2relu_delay_{DELAY_STEPS}_model.pt"


    if TRAIN:
        model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
        criterion = nn.MSELoss()

        train_sequences = load_bags("datasets/rzr_real", reader)
        train_dataset = LookaheadDataset(train_sequences, ["control", "state"], ["target"], delay_steps=DELAY_STEPS)

        val_dataset = LookaheadDataset(val_sequences, ["control", "state"], ["target"], delay_steps=DELAY_STEPS)

        train(
            model,
            optimizer,
            DataLoader(train_dataset, batch_size=32, shuffle=True),
            criterion,
            epochs=EPOCHS,
            val_loader=DataLoader(val_dataset, batch_size=32, shuffle=True),
            verbose=True,
        )

        torch.jit.script(model).save(model_name)
    else:
        model = torch.jit.load(model_name)

    if PLOT_VAL:
        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            # Take the longest sequence from validation set
            longest_val_sequence = val_sequences[np.argmax(np.array([len(s[list(s.keys())[0]]) for s in val_sequences]))]

            y_true_all = []
            y_zero_all = []
            y_pred_all = []
            dts_all = []
            pred_acc_all = []


            for x, y, dt in tqdm(DataLoader(LookaheadDataset([longest_val_sequence], ["control", "state"], ["target"], delay_steps=DELAY_STEPS), batch_size=1, shuffle=False), desc="Final Validation"):
                pred_acc = model(x)
                y_pred = pred_acc * dt + x[:, -2:]
                y_zero_all.extend(x[:, -2:].detach().cpu().numpy())
                y_pred_all.extend(y_pred.detach().cpu().numpy())
                y_true_all.extend(y.detach().cpu().numpy())
                pred_acc_all.extend(pred_acc.detach().cpu().numpy())
                dts_all.extend(dt.detach().cpu().numpy())

            pred_acc_all = np.array(pred_acc_all)
            v_disp_pred_all = pred_acc_all * np.array(dts_all)

            # Plot histogram of predicted displacement
            plt.hist2d(v_disp_pred_all[:, 0], v_disp_pred_all[:, 1], bins=100, cmap="Blues")
            plt.xlabel("dx (m/s)")
            plt.ylabel("dtheta (rad./s)")
            plt.title("Predicted velocity Displacement (acc./model output * dt)")
            if not os.path.exists("plots"):
                os.makedirs("plots")
            plt.savefig(f"plots/dx_dtheta_pred_delay_{DELAY_STEPS}.png")
            plt.show()

            # Convert from (dx, dtheta) to (dx, dy, dtheta)
            y_true_all = np.array(y_true_all)
            tmp = np.zeros((len(y_true_all), 3))
            tmp[:, 0] = y_true_all[:, 0]
            tmp[:, 2] = y_true_all[:, 1]
            y_true_all = tmp

            y_zero_all = np.array(y_zero_all)
            tmp = np.zeros((len(y_zero_all), 3))
            tmp[:, 0] = y_zero_all[:, 0]
            tmp[:, 2] = y_zero_all[:, 1]
            y_zero_all = tmp

            y_pred_all = np.array(y_pred_all)
            tmp = np.zeros((len(y_pred_all), 3))
            tmp[:, 0] = y_pred_all[:, 0]
            tmp[:, 2] = y_pred_all[:, 1]
            y_pred_all = tmp

            dts_all = np.array(dts_all)

            # We will only unroll continuous sequence.
            dts_all = dts_all[::DELAY_STEPS]
            y_true_all = y_true_all[::DELAY_STEPS]
            y_zero_all = y_zero_all[::DELAY_STEPS]
            y_pred_all = y_pred_all[::DELAY_STEPS]

            # FIXME: Commented out code below is incorrect, since it doesn't unroll the acceleration
            # # Plot the dx, dtheta for each configuration
            # fig, axs = plt.subplots(3, 2, figsize=(10, 10))
            # for row_idx, (y, type_y) in enumerate(zip([y_true_all, y_zero_all, y_pred_all], ["True", "Zero", "Pred"])):
            #     # for col_idx, col_name in enumerate(["dx", "dtheta"]):
            #     #     axs[row_idx, col_idx].plot(np.cumsum(dts_all), y[:, col_idx])
            #     #     axs[row_idx, col_idx].set_title(f"{type_y} - {col_name}")
            #     for col_idx, (y_idx, col_name) in enumerate(zip([0, 2], ["dx", "dtheta"])):
            #         axs[row_idx, col_idx].plot(np.cumsum(dts_all), y[:, y_idx])
            #         axs[row_idx, col_idx].set_title(f"{type_y} - {col_name}")
            # plt.show()

            # For Zero and Predicted plot rollouts only for PLOT_LEN_ROLLOUT seconds
            idx = 0
            ts = np.cumsum(dts_all)

            poses_true = reconstruct_poses_from_odoms(y_zero_all, dts_all)

            # plt.plot(poses_true[:, 0], poses_true[:, 1], label="True")
            plt.plot(poses_true[:, 0], poses_true[:, 1], label="True")
            while idx * PLOT_LEN_ROLLOUT < ts[-1]:
                cur_idxs = np.where(ts // PLOT_LEN_ROLLOUT == idx)[0]

                poses_first_idx = cur_idxs[0]

                start_vel = np.array([np.cos(poses_true[poses_first_idx, 2]) * y_zero_all[cur_idxs[0], 0], np.sin(poses_true[poses_first_idx, 2]) * y_zero_all[cur_idxs[0], 0], y_zero_all[cur_idxs[0], 2]])

                poses_pred = reconstruct_poses_from_acc(pred_acc_all[cur_idxs], dts_all[cur_idxs], start_vel=start_vel, start_pose=poses_true[poses_first_idx])
                poses_zero = reconstruct_poses_from_acc(np.zeros((len(cur_idxs), 3)), dts_all[cur_idxs], start_vel=start_vel, start_pose=poses_true[poses_first_idx])

                if poses_zero is None or poses_pred is None:
                    idx += 1
                    continue

                if idx == 0:
                    plt.plot(poses_zero[:, 0], poses_zero[:, 1], color="gray", label="Zero")
                    plt.plot(poses_pred[:, 0], poses_pred[:, 1], color="orange", label="Pred")
                else:
                    plt.plot(poses_zero[:, 0], poses_zero[:, 1], color="gray")
                    plt.plot(poses_pred[:, 0], poses_pred[:, 1], color="orange")

                idx += 1
            plt.legend()
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            if not os.path.exists("plots"):
                os.makedirs("plots")
            plt.savefig(f"plots/hidden2relu_delay_{DELAY_STEPS}_rollout.png")
            plt.show()


if __name__ == "__main__":
    main()
