import numpy as np
from rosbag2torch import readers, filters, load_bags
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    DATASET_TRAIN = "datasets/rzr_real_backward"
    dataset_name = DATASET_TRAIN.split("/")[-1]

    features = ["control", "state"]
    delayed_features = ["target"]

    # # Async Reader
    train_reader = readers.ASyncSequenceReader(
        list(set(features + delayed_features)),
        features_to_record_on=["control"],
        filters=[
            filters.BackwardFilter(),
            filters.PIDInfoFilter()
        ]
    )

    # Fixed Timestamp Reader
    # log_hz = 30
    # train_reader = readers.FixedIntervalReader(
    #     list(set(features + delayed_features)),
    #     log_interval=1.0 / log_hz,
    #     filters=[
    #         filters.ForwardFilter(),
    #         filters.PIDInfoFilter()
    #     ]
    # )

    train_sequences = load_bags(DATASET_TRAIN, train_reader)

    # print(train_sequences)

    state_dx = []
    state_dtheta = []

    control_steer = []
    control_throttle = []
    control_brake = []

    for sequence in train_sequences:
        state_dx.extend(sequence["state"][:, 0])
        state_dtheta.extend(sequence["state"][:, 1])

        control_throttle.extend(sequence["control"][:, 0])
        control_brake.extend(sequence["control"][:, 1])
        control_steer.extend(sequence["control"][:, 2])

    control_throttle_brake_combined = np.array(control_throttle) - np.array(control_brake)

    if not Path(f"plots/{dataset_name}").exists():
        Path(f"plots/{dataset_name}").mkdir()

    for title, data in [
        ("state_dx", state_dx),
        ("state_dtheta", state_dtheta),
        ("control_steer", control_steer),
        ("control_throttle", control_throttle),
        ("control_brake", control_brake),
        ("control_throttle_brake_combined", control_throttle_brake_combined),
    ]:
        plt.figure()
        plt.title(f"{title} - {np.mean(data):.4g}")
        plt.hist(data)
        plt.savefig(f"plots/{dataset_name}/hist_{title}.png", bbox_inches="tight")

        plt.cla()
        plt.clf()

    for label_x, data_x, label_y, data_y in [
        ("Throttle", control_throttle, "Brake", control_brake),
    ]:
        plt.figure()
        plt.hist2d(data_x, data_y)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.savefig(f"plots/{dataset_name}/hist2d_{label_x}_{label_y}.png", bbox_inches="tight")

        plt.cla()
        plt.clf()


if __name__ == "__main__":
    main()
