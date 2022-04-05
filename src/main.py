def plot_preds_vs_gt(model, dataloader):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    total_y_true = []
    total_y_pred = []
    total_ts = []

    with torch.no_grad():
        for x, y, t in dataloader:
            y_pred = model(x)
            y_pred *= t  # Acceleration
            total_y_true.extend(y.numpy())
            total_y_pred.extend(y_pred.numpy())
            total_ts.extend(t.numpy())

    total_y_pred = np.array(total_y_pred)
    total_y_true = np.array(total_y_true)
    total_ts = np.array(total_ts)

    np.save("data/y_pred.npy", total_y_pred)
    np.save("data/y_true.npy", total_y_true)
    np.save("data/ts.npy", total_ts)

    def unroll_y(y: np.ndarray, t: np.ndarray):
        thetas = np.cumsum(y[:, 1] * t)

        along_vecs = []

        for theta, vx, dt in zip(thetas, y[:, 0], t):
            along_vec = np.array([np.cos(theta), np.sin(theta)])
            along_vec /= np.linalg.norm(along_vec)
            along_vec *= vx * dt
            along_vecs.append(along_vec)

        return np.cumsum(np.array(along_vecs), axis=0)


    total_y_pred = unroll_y(total_y_pred, total_ts)
    total_y_true = unroll_y(total_y_true, total_ts)

    plt.plot(total_y_pred[:, 0], total_y_pred[:, 1], label="Pred")
    plt.plot(total_y_true[:, 0], total_y_true[:, 1], label="True")
    plt.legend()
    plt.show()


def main():
    spec = parse_args()

    model = load_model(spec["model"])

    data = load_dataset(
        dataset_name=spec["dataset"],
        x_features=model.x_features,
        y_features=model.y_features,
        robot_type=spec["robot"],
        dataset_type=model.dataset_name,
    )

    if model.dataset_name == "numpy":
        model = train_numpy(data, model)
    elif model.dataset_name == "torch_lookahead":
        import torch
        from torch import optim, nn
        from torch.utils.data import DataLoader
        dl = DataLoader(data, batch_size=32, shuffle=False)
        model = model()

        train_torch_simple(model, optim.Adam(model.parameters()), dl, nn.MSELoss(), 200)

        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save('model_scripted.pt') # Save

        plot_preds_vs_gt(model, dl)


if __name__ == "__main__":
    from parse_args import parse_args
    from data_utils import load_dataset
    from models import load_model
    from optimization_logic import train_numpy, train_torch_simple

    main()
