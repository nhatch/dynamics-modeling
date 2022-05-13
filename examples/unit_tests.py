import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from example_utils import get_world_frame_rollouts
import torch

class Tests:
    @staticmethod
    def straight_line(model: torch.nn.Module):
        """Drives vehicle in a straight line, with a constant throttle.

        NOTE: Plot might appear curved a lot
        """

        def __curvature(x: np.ndarray, y: np.ndarray):
            dx = np.gradient(x)
            dy = np.gradient(y)
            v = np.vstack((dx, dy)).T

            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            a = np.vstack((ddx, ddy)).T

            return np.cross(v, a) / (np.linalg.norm(v, ord=1, axis=1) ** 3)

        dt = 1 / 30  # seconds
        rollout_s = 5  # seconds
        throttle_val = 0.4

        num_samples = int(rollout_s / dt)

        controls = torch.zeros((num_samples, 3)).float()
        states = torch.zeros((num_samples, 2)).float()
        dts = torch.full((num_samples,), dt).float()

        controls[:, 0] = throttle_val

        _, _, poses_pred = get_world_frame_rollouts(model, states, controls, dts, rollout_s)
        poses_pred = np.array(poses_pred).squeeze()

        k = __curvature(poses_pred[:, 0], poses_pred[:, 1])
        max_k = np.max(np.abs(k))

        plt.title(f"Straight line - $|\kappa|_{{max}} = {max_k:.3g}$")
        plt.plot(poses_pred[:, 0], poses_pred[:, 1])
        plt.show()


    @staticmethod
    def circle(model: torch.nn.Module):
        """Drives vehicle in a circle, with a small constant throttle, and max steering.

        NOTE: Circle might not be fully closed.
            Radius is calculated as half the distance from origin to the furthest point on the circle.
        """
        dt = 1 / 30  # seconds
        rollout_s = 14  # seconds
        throttle_val = 0.15

        num_samples = int(rollout_s / dt)

        controls = torch.zeros((num_samples, 3)).float()
        states = torch.zeros((num_samples, 2)).float()
        dts = torch.full((num_samples,), dt).float()

        controls[:, 0] = throttle_val
        controls[:, 2] = 1

        _, _, poses_pred = get_world_frame_rollouts(model, states, controls, dts, rollout_s)
        poses_pred = np.array(poses_pred).squeeze()

        # Find radius
        radius = np.max(np.linalg.norm(poses_pred[:, :2], axis=1)) / 2

        plt.title(f"Circle - $r = {radius:.4g}$")
        plt.plot(poses_pred[:, 0], poses_pred[:, 1])
        plt.show()

    @staticmethod
    def run_test(model: torch.nn.Module, test_name: str):
        # Get availale tests, filter built in values
        available_tests = vars(Tests)
        available_tests = dict(filter(lambda x: not (x[0].startswith("__") or x[0] == "run_test"), available_tests.items()))

        if test_name not in available_tests:
            print(f"Test with name: {test_name} not found.")
            print(f"Available tests are: {', '.join(available_tests.keys())}")

        available_tests[test_name].__func__(model)


def main():
    parser = argparse.ArgumentParser("unit_test")
    parser.add_argument("model", type=str)
    parser.add_argument("test_name", type=str)

    args = parser.parse_args()

    print(args.model)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model does not exist at path: {model_path}")
        exit(1)

    model = torch.jit.load(str(model_path))

    Tests.run_test(model, args.test_name)


if __name__ == "__main__":
    main()
