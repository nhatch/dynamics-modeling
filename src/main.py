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
    specs = parse_args()

    for spec in specs:
        model = load_model(spec["model"]["type"])

        train_data = load_dataset(
            dataset_name=spec["dataset"]["train_name"],
            config=spec,
        )


        if model.dataset_name == "numpy":
            model = train_numpy(train_data, model)
        elif model.dataset_name == "torch_lookahead":
            import torch
            from torch import optim, nn
            from torch.utils.data import DataLoader

            if "val_name" in spec["dataset"]:
                val_data = load_dataset(
                    dataset_name=spec["dataset"]["val_name"],
                    config=spec,
                )
                val_dl = DataLoader(val_data, batch_size=32, shuffle=False)

            dl = DataLoader(train_data, batch_size=32, shuffle=False)
            model = model()

            if spec["train_loop"]["type"] == "regression_train_loop":
                train_torch_simple(
                    model,
                    optim.Adam(model.parameters()),
                    dl,
                    nn.MSELoss(),
                    **spec["train_loop"]["args"],
                    val_loader=val_dl if "val_dl" in vars() else None,
                )
            else:
                raise ValueError()

            if "save_prefix" in spec["model"]:
                save_prefix = spec["model"].get("save_prefix")
                model_scripted = torch.jit.script(model) # Export to TorchScript
                model_scripted.save(f'{save_prefix}.pt') # Save

            plot_preds_vs_gt(model, dl)


if __name__ == "__main__":
    from command_line.parse_args import parse_args
    from command_line.data_utils import load_dataset
    from command_line.models import load_model
    from command_line.optimization_logic import train_numpy, train_torch_simple

    main()
