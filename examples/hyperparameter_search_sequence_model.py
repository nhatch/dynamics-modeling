from datetime import datetime
import numpy as np

from rosbag2torch import filters, load_bags, readers
import torch
from typing import Tuple
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from example_utils import augment_sequences_reflect_steer, unroll_sequence_torch, single_hyperparameter_thread


def loss_from_batch(model: nn.Module, batch: Tuple[torch.Tensor, ...], criterion: nn.Module = nn.MSELoss()) -> torch.Tensor:
    # Unpack batch
    controls, states, targets, dts = batch

    # Convert to FloatTensor
    controls, states, targets, dts = controls.float(), states.float(), targets.float(), dts.float()

    # Forward pass - Unroll the trajectory
    predictions = unroll_sequence_torch(
        model=model,
        start_state=states[:, 0],
        controls=controls,
        dts=dts
    )

    # At each point of trajectory, calculate the loss
    rollout_losses = []
    for pred, target in zip(predictions, targets.transpose(0, 1)):
        loss = criterion(pred, target)
        rollout_losses.append(loss)

    # Total loss is the sum of the losses at each trajectory point
    loss = sum(rollout_losses)

    return loss


def main():
    DATASET_TRAIN = "datasets/rzr_real"
    DATASET_VAL = "datasets/rzr_real_val"

    # What features to read.
    # NOTE: Values corresponding to these (and their ordering) are hardcoded in the model and train loop. Changing them here will break the code.
    features = ["control", "state"]
    delayed_features = ["target"]

    # Fixed Timestamp Reader
    log_hz = 30
    fixed_interval_reader = readers.FixedIntervalReader(
        list(set(features + delayed_features)),
        log_interval=1.0 / log_hz,
        filters=[
            filters.ForwardFilter(),
            filters.PIDInfoFilter()
        ]
    )
    # Async Reader
    async_reader = readers.ASyncSequenceReader(
        list(set(features + delayed_features)),
        features_to_record_on=["control"],
        filters=[
            filters.ForwardFilter(),
            filters.PIDInfoFilter()
        ]
    )

    # Train sequences
    fixed_interval_sequences = load_bags(DATASET_TRAIN, fixed_interval_reader)
    async_sequences = load_bags(DATASET_TRAIN, async_reader)

    # Augment train sequences
    fixed_interval_sequences = augment_sequences_reflect_steer(fixed_interval_sequences)
    async_sequences = augment_sequences_reflect_steer(async_sequences)

    # Validation Sequences
    val_sequences = load_bags(DATASET_VAL, async_reader)

    date_suffix = datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/hyperparameter_search_sequence_model_{date_suffix}")

    # MSE for everything
    criterion = nn.MSELoss()
    forward_fn = lambda *args, **kwargs: loss_from_batch(*args, **kwargs, criterion=criterion)

    NUM_HYPERPARAM_SEARCHES = 50
    for _ in range(NUM_HYPERPARAM_SEARCHES):
        # Sequence/Data Parameters
        delay_steps = np.random.randint(3, 8)  # indices
        # delay_steps = 7  # indices
        # rollout_s = 8  # seconds
        rollout_s = np.random.randint(1, 9)  # seconds

        # Model Parameters
        activation_fn = np.random.choice([nn.SELU(), nn.ReLU(), nn.LeakyReLU(), nn.Tanh()])
        hidden_size = int(2 ** np.random.randint(3, 8))
        num_hidden_layers = np.random.randint(1, 4)
        collapse_throttle_brake = bool(np.random.choice([True, False]))

        # Training Parameters
        epochs = 50  # Epochs are fixed
        lr = 10 ** np.random.uniform(-5, -2)
        batch_size = int(2 ** np.random.randint(6, 11))

        reader_str = np.random.choice(["fixed_interval", "async"])

        if reader_str == "fixed_interval":
            train_sequences = fixed_interval_sequences
        else:
            train_sequences = async_sequences

        hyperparam_settings = {
            "delay_steps": delay_steps,
            "rollout_s": rollout_s,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "batch_size": batch_size,
            "activation_fn": activation_fn,
            "lr": lr,
            "reader_str": reader_str,
            "collapse_throttle_brake": collapse_throttle_brake
        }

        hyperparam_writer_path, hyperparam_val_loss = single_hyperparameter_thread(
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            features=features,
            delayed_features=delayed_features,
            log_hz=log_hz,
            epochs=epochs,
            forward_fn=forward_fn,
            model_prefix="sequence_model",
            **hyperparam_settings
        )

        # For writing to tensorboard we want activation function to be a string
        hyperparam_settings["activation_fn"] = hyperparam_settings["activation_fn"].__class__.__name__

        writer.add_hparams(
            hparam_dict=hyperparam_settings,
            metric_dict={
                "val_loss": hyperparam_val_loss
            },
            run_name=hyperparam_writer_path
        )


if __name__ == "__main__":
    main()
