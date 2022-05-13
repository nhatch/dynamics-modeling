from collections import defaultdict
import copy
import torch
from torch import nn
from torch import optim
from datetime import datetime
import numpy as np
from typing import Optional, Tuple, List, Callable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

from rosbag2torch.bag_processing.sequence_readers.abstract_sequence_reader import Sequences
from rosbag2torch.datasets.torch_sequence_lookahead import SequenceLookaheadDataset

def reconstruct_poses_from_odoms(d_odom: np.ndarray, dt: np.ndarray, start_pose: Optional[np.ndarray] = None):
    """This function reconstructs the trajectory from odometry data.
    Odometry data is assumed to be in the form of [v_x, v_y, v_theta] or [v_x, v_theta],
    with dx and dy being in the robot frame.

    Args:
        d_odom (np.ndarray): A (n, 2) or (n, 3) array of odometry data.
        dt (np.ndarray): A (n,) array of differences in timestamps.
            It needs take delay_steps into account.
            This means that: dt[i] = t[i + delay_steps] - t[i]
        start_pose (Optional[np.ndarray], optional): A (3,) array of the starting pose (x, y, theta).
            Defaults to (0, 0, 0).

    Returns:
        np.ndarray: An (n + 1, 3) array of poses.
            Row i + 1 is the pose after d_odom[i] has been applied.
            The first row is the start pose.
            Columns correspond to (x, y, theta)
    """
    # Asserts to double check that everything is in the right format
    assert len(d_odom.shape) == 2 and d_odom.shape[1] in {2, 3}, \
        f"d_odom must be a 2D array with 2 (dx, dtheta) or 3 (dx, dy, dtheta) columns. Instead it is of shape {d_odom.shape}"
    # Default value for start_pose + assert if specified
    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])
    else:
        assert len(start_pose.shape) == 1 and start_pose.shape[0] == 3, \
            f"If start_pose is specified it must be a 1D array of length 3. Instead it is of shape {start_pose.shape}"

    # If d_odom has 2 columns add a column of zeros in the middle (for dy)
    if d_odom.shape[1] == 2:
        tmp = np.zeros((len(d_odom), 3))
        tmp[:, [0, 2]] = d_odom
        d_odom = tmp

    # We expect dt to be a 1D array. It may be a (n, 1) array, in which case we'll reshape it to (n,).
    dt = dt.squeeze()


    # Unroll thetas first. This is because dx, dy are dependent on value of theta at each step.
    thetas = np.cumsum(d_odom[:, 2] * dt, axis=0) + start_pose[2]

    # Create vectors along and orthogonal to theta
    along_vec = np.concatenate((np.cos(thetas)[..., None], np.sin(thetas)[..., None]), axis=1)
    # Orthogonal vector is -sin(theta) along x and cos(theta) along y, so we can just use along
    ortho_vec = along_vec[..., [1, 0]]
    ortho_vec[..., 0] *= -1

    # Unroll the poses
    poses = start_pose[:2] + np.cumsum(dt[..., None] * (along_vec * d_odom[:, 0, None] + ortho_vec * d_odom[:, 1, None]), axis=0)

    result = np.hstack((poses, thetas[:, None]))
    result = np.vstack((start_pose[None, :], result))

    return result


def reconstruct_poses_from_acc(acc: np.ndarray, dt: np.ndarray, start_pose: Optional[np.ndarray] = None, start_vel: Optional[np.ndarray] = None):
    if start_pose is None:
        start_pose = np.zeros(3)
    if start_vel is None:
        start_vel = np.zeros(3)

    # Convert from (x'', theta'') to (x'', y'', theta'')
    if acc.shape[1] == 2:
        tmp = np.zeros((len(acc), 3))
        tmp[:, [0, 2]] = acc
        acc = tmp

    # dt can sometimes be (n, 1), so just for sanity check we'll make sure it's (n,).
    dt = dt.squeeze()

    # 1. Rollout thetas
    disp_v_theta = np.cumsum(acc[:, 2] * dt)
    theta_cum = np.cumsum((start_vel[2] + disp_v_theta) * dt)
    thetas = (theta_cum + start_pose[2]).squeeze()

    # 2. Rollout velocities
    along_vec = np.concatenate((np.cos(thetas)[..., None], np.sin(thetas)[..., None]), axis=1)
    ortho_vec = along_vec[..., [1, 0]]

    # 3. Rollout velocities
    disp_v = np.cumsum(dt[..., None] * (along_vec * acc[:, 0, None] + ortho_vec * acc[:, 1, None]), axis=0) + start_vel[None, :2]
    disp_v = np.hstack((disp_v, disp_v_theta[:, None]))

    v = disp_v + start_vel

    return np.cumsum(dt[..., None] * v, axis=0) + start_pose


class StateControlBaseline(nn.Module):
    def __init__(self, dt: float, min_linear_vel: float, forward_force_multiplier: float):
        """Baseline model for state-control input.

        Args:
            dt (float): Difference in time between consecutive poses in a single rollout.
            min_linear_vel (float): Minimum linear velocity to be considered.
            forward_force_multiplier (float): Multiplier for the throttle.
        """
        super().__init__()
        self.dt = dt
        self.min_linear_vel = min_linear_vel
        # maximum accel attainable through max throttle/break input
        self.forward_force_multiplier = forward_force_multiplier

    def forward(self, control: torch.Tensor, state: torch.Tensor):
        linear_vel = state[:, 0]
        angular_vel = state[:, 1]
        forward_force = control[:, 0]
        steering = control[:, 2]

        linear_accel = torch.maximum(
            forward_force * self.forward_force_multiplier,
            (-linear_vel + self.min_linear_vel) / self.dt)
        angular_accel = (steering * linear_vel - angular_vel) / self.dt

        return torch.stack((linear_accel, angular_accel), dim=1)


class StateControlTrainableModel(nn.Module):
    # jit stuff
    collapse_throttle_brake: torch.jit.Final[bool]

    def __init__(self, activation: nn.Module, hidden_size: int, num_hidden_layers: int, collapse_throttle_brake: bool = False) -> None:
        super().__init__()
        assert hidden_size > 0
        assert num_hidden_layers >= 0

        self.collapse_throttle_brake = bool(collapse_throttle_brake)

        dim_in = 4 if collapse_throttle_brake else 5

        # Edge case: no hidden layers
        if num_hidden_layers == 0:
            self._model = nn.Linear(dim_in, 2)
        else:
            self._model = nn.Sequential(
                nn.Linear(dim_in, hidden_size),
                activation,
                *[
                    nn.Sequential(nn.Linear(hidden_size, hidden_size), activation)
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_size, 2)
            )

    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Control is throttle, brake, steer
        if self.collapse_throttle_brake:
            control[:, 0] = control[:, 0] - control[:, 1]
            input_control = torch.cat([control[:, 0, None], control[:, 2, None]], dim=1)
        else:
            input_control = control
        return self._model(torch.cat((input_control, state), dim=1))


def augment_sequences_reflect_steer(sequences: Sequences) -> Dataset:
    """Augment sequences by reflecting the steering angle."""
    num_sequences = len(sequences)  # Append to sequences in-place. We want to only iterate over the original sequences.
    for i in range(num_sequences):
        sequence = {}
        for feature_name, feature_vals in sequences[i].items():
            if feature_name in {"state", "target", "control"}:
                feature_vals[..., -1] *= -1
            sequence[feature_name] = feature_vals

        sequences.append(sequence)

    return sequences


def unroll_sequence_torch(
    model: nn.Module,
    start_state: torch.Tensor,
    controls: torch.Tensor,
    dts: torch.Tensor,
) -> List[torch.Tensor]:
    """ Unroll the model forward for a sequence of states.
    In this function both states in and out are in robot frame.

    Args:
        model (nn.Module): Model to use for prediction of each step.
        start_state (torch.Tensor): State to start rollout from.
            Shape: (N, *), where * is the state shape
        controls (torch.Tensor): Controls to be applied at each step of the rollout.
            (N, S, *), where N is the batch size, S is the sequence length, and * is the control shape.
        dts (torch.Tensor): Difference in time between each step of the rollout.
            (N, S), where N is the batch size, and S is the sequence length

    Returns:
        List[torch.Tensor]: A List of length S, where each element is a (N, *) Tensor, where * is the state shape
    """
    controls = controls.transpose(0, 1)
    dts = dts.transpose(0, 1)

    cur_state = start_state
    result = []
    for control, dt in zip(controls, dts):
        # acceleration * dt + prev state
        # m / s^2 * s + m / s
        cur_state = model(control=control, state=cur_state) * dt.view(-1, 1) + cur_state

        result.append(cur_state)
    return result


def get_world_frame_rollouts(model: nn.Module, states: torch.Tensor, controls: torch.Tensor, dts: torch.Tensor, rollout_in_seconds: float) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Converts a sequence of robot frame states to world frame.
    It then uses a model to rollout the trajectory of length rollout_in_seconds in world frame, for each such interval in sequence.

    For example, if sequence if 10s long and rollout_in_seconds is 4s then this function will return:
        - Continuous sequence of true poses of length 10s
        - Start poses of each of predicted sequences (see below), corresponding to true poses at times 0s, 4s, 8s
        - Three continuos sequences of predicted poses of lengths 4s, 4s, 2s one for each interval in sequence,
            each starting at corresponding start pose.

    Args:
        model (nn.Module): Model to unroll the sequence with.
        states (torch.Tensor): (N, *) Tensor of robot frame states.
        controls (torch.Tensor): (N, *) Tensor of controls applied at each state.
        dts (torch.Tensor): (N, *) Tensor of time steps between each two consecutive states.
        rollout_in_seconds (float): Length of each rollout in seconds.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray]]: Tuple of:
            - Continuous sequence of true poses for the whole sequence of length N + 1 (first pose will always be 0)
            - Start poses of each of predicted sequences (see below),
                corresponding to true poses at times 0s, rollout_in_seconds, 2*rollout_in_seconds, ...
            - List of predicted sequences, each of length rollout_in_seconds, starting at corresponding start pose.
    """
    controls, states, dts = controls.float(), states.float(), dts.float()

    # Controls need to be reshaped to also have batch size
    controls = controls.view(1, *controls.shape)
    dts = dts.view(1, *dts.shape)

    # Unroll the true trajectory
    poses_true = reconstruct_poses_from_odoms(states.numpy(), dts.numpy())

    # Get timestamp relative to the start of the rollout
    ts = np.cumsum(dts.numpy()) - dts.numpy()[0]

    # Iterate through the rollout in chunks of length ROLLOUT_S (in seconds)
    idx = 0  # index of the current rollout chunk
    start_poses = []  # poses at the start of each rollout chunk

    poses_pred_all = []

    while idx * rollout_in_seconds <= ts[-1]:
        # Get the current rollout chunk
        cur_idx = np.where(ts // rollout_in_seconds == idx)[0]

        # Get the corresponding dts, controls and states
        cur_dts = dts[:, cur_idx]
        cur_controls = controls[:, cur_idx]
        cur_states = states[cur_idx]

        # Unroll the model predictions for the current rollout chunk
        predictions = unroll_sequence_torch(
            model=model,
            start_state=cur_states[None, 0],
            controls=cur_controls,
            dts=cur_dts
        )

        # Convert to numpy
        np_predictions = np.array(
            [pred.detach().cpu().numpy() for pred in predictions]
        ).squeeze()

        # Convert (dx, dtheta) to (x, y, theta)
        poses_pred = reconstruct_poses_from_odoms(np_predictions, cur_dts.numpy(), start_pose=poses_true[cur_idx[0]])

        # Save the start pose for this chunk
        start_poses.append(poses_true[cur_idx[0]])

        poses_pred_all.append(poses_pred)
        idx += 1

    return poses_true, np.array(start_poses), poses_pred_all


def train(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, Tuple[torch.Tensor, ...]], torch.Tensor],
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epochs: int,
    val_loader: Optional[DataLoader] = None,
    model_baseline: Optional[nn.Module] = None,
    verbose: bool = True,
    writer: Optional[SummaryWriter] = None,
):
    best_val_loss = np.inf
    best_state_dict = None
    best_state_dict = None

    trange_epochs = trange(epochs, desc="Epochs", disable=not verbose, leave=True)
    for epoch in trange_epochs:
        running_total_loss = 0.0
        running_baseline_loss = 0.0

        for batch in tqdm(train_loader, disable=not verbose, desc="Train", leave=False):
            # Zero-out the gradient
            optimizer.zero_grad()

            # Forward pass + Calculation of loss
            loss = forward_fn(model, batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log loss
            running_total_loss += loss.detach().cpu().item()

            # Repeat for baseline model
            with torch.no_grad():
                if model_baseline is not None:
                    baseline_loss = forward_fn(model_baseline, batch)

                    running_baseline_loss += baseline_loss.detach().cpu().item()

        # Run zero_grad at the end of each epoch, just in case
        optimizer.zero_grad()

        train_loss = running_total_loss / len(train_loader)
        train_baseline_loss = running_baseline_loss / len(train_loader)
        if writer is not None:
            writer.add_scalar("Loss/train total", train_loss, epoch)
            if model_baseline is not None:
                writer.add_scalar("Loss/train total baseline", train_baseline_loss, epoch)
        desc = f"Epochs Train Loss {train_loss:.4g} Baseline {train_baseline_loss:.4g}"

        if val_loader is not None:
            with torch.no_grad():
                running_loss = 0.0

                running_baseline_loss = 0.0

                for batch in tqdm(val_loader, disable=not verbose, desc="Val", leave=False):

                    loss = forward_fn(model, batch)
                    running_loss += loss.detach().cpu().item()

                    # Repeat for baseline model
                    if model_baseline is not None:
                        baseline_loss = forward_fn(model_baseline, batch)
                        running_baseline_loss += baseline_loss.detach().cpu().item()

                val_loss = running_loss / len(val_loader)
                val_baseline_loss = running_baseline_loss / len(val_loader)
                if writer is not None:
                    writer.add_scalar("Loss/val total", val_loss, epoch)
                    if model_baseline is not None:
                        writer.add_scalar("Loss/val total baseline", val_baseline_loss, epoch)
                desc += f" Val Loss {val_loss:.4g} Baseline {val_baseline_loss:.4g}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = copy.deepcopy(model.state_dict())

        trange_epochs.set_description(desc)

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return best_val_loss


def plot_rollout(model: nn.Module, dataset: SequenceLookaheadDataset, rollout_s: float) -> np.ndarray:
    """Given a dataset plots the longest sequence from it.
    Each rollout using model in that sequence will be of at most `rollout_s` length (in second).

    Args:
        model (nn.Module): Model to use for rolling out states
        dataset (SequenceLookaheadDataset): Dataset to get the longest sequence from
        rollout_s (float): Number of seconds each rollout should last.

    Returns:
        np.ndarray: Numpy array respresenting image with model rollouts on it as well as the true trajectory of robot.

    Note:
        - This code uses matplotlib to render canvas, which is slow.
            It's sufficient for once per epoch or at the end of training plotting,
            but for more frequent plotting you might want to use cv2.
    """
    fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(111)
    ax: plt.Axes

    controls, _, states, states_dts, targets, target_dts = dataset.longest_rollout
    dts = target_dts - states_dts
    poses_true, start_poses, poses_pred_all = get_world_frame_rollouts(model, states, controls, dts, rollout_in_seconds=rollout_s)

    ax.scatter(start_poses[:, 0], start_poses[:, 1], color="red", marker="X")
    ax.plot(poses_true[:, 0], poses_true[:, 1], color="red", label="True")
    for idx, poses_pred in enumerate(poses_pred_all):
        plt_kwargs = {"color": "black"}
        if idx == 0:
            plt_kwargs["label"] = "Predicted"
        ax.plot(poses_pred[:, 0], poses_pred[:, 1], **plt_kwargs)

    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    #Image from plot
    ax.axis('off')
    fig.tight_layout(pad=0)

    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return image_from_plot


def single_hyperparameter_thread(
    train_sequences: Sequences,
    val_sequences: Sequences,
    features: List[str],
    delayed_features: List[str],
    delay_steps: float,
    rollout_s: float,
    hidden_size: float,
    num_hidden_layers: float,
    epochs: float,
    batch_size: float,
    activation_fn: nn.Module,
    lr: float,
    reader_str: str,
    log_hz: float,
    collapse_throttle_brake: bool,
    forward_fn: Callable[[nn.Module, Tuple[torch.Tensor, ...]], torch.Tensor],
    model_prefix: str = "sequence_model",
    rollout_s_validation: float = 5.0,
):
    rollout_len = int(rollout_s  * log_hz)

    train_dataset = SequenceLookaheadDataset(train_sequences, [("control", 0), ("state", delay_steps), ("target", delay_steps + 1)], sequence_length=rollout_len)
    val_dataset = SequenceLookaheadDataset(val_sequences, [("control", 0), ("state", delay_steps), ("target", delay_steps + 1)], sequence_length=int(rollout_s_validation * log_hz))
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        return None

    settings_suffix = f"delay_{delay_steps}_rollout_{rollout_s}s_hidden_{hidden_size}_layers_{num_hidden_layers}_activation_{activation_fn.__class__.__name__}_lr_{lr:.2e}_bs_{batch_size}_epochs_{epochs}_reader_{reader_str}"
    date_suffix = datetime.now().strftime("%b%d_%H-%M-%S")

    model = StateControlTrainableModel(activation=activation_fn, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, collapse_throttle_brake=collapse_throttle_brake)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01, lr=lr)

    writer_name = f"runs/{model_prefix}_{settings_suffix}_{date_suffix}"
    nn_module_prefix = f"models/{model_prefix}_{settings_suffix}_{date_suffix}"

    writer = SummaryWriter(log_dir=writer_name)

    val_loss = train(
        model=model,
        optimizer=optimizer,
        forward_fn=forward_fn,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        model_baseline=StateControlBaseline(delay_steps * 1.0 / log_hz, 0.001, 3.0),
        epochs=epochs,
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
        verbose=True,
        writer=writer
    )

    torch.save(model.state_dict(), f"{nn_module_prefix}_state_dict.pt")
    torch.jit.script(model).save(f"{nn_module_prefix}_scripted.pt")

    train_img = plot_rollout(model, train_dataset, rollout_s_validation)
    writer.add_image("Best Model (Train Sequence)", train_img, 0, dataformats="HWC")

    val_img = plot_rollout(model, val_dataset, rollout_s_validation)
    writer.add_image("Best Model (Val Sequence)", val_img, 0, dataformats="HWC")

    return writer_name, val_loss
