from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import trange, tqdm


def evaluate(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    loader: DataLoader,
    device: str = "cpu",
    verbose: bool = True,
):
    with torch.no_grad():
        running_loss = 0.0
        for x, y, t in tqdm(loader, disable=not verbose, leave=False):
            x, y, t = x.to(device), y.to(device), t.to(device)
            y_pred = model(x) * t
            running_loss += criterion(y_pred, y).cpu().item()

    return running_loss / len(loader)


# TODO: Implement these two functions.
# We might want to include some of this functionality within the nn.Module itself (i.e. subclass it)
# This way the torchscripted module would also contain that functionality and this might make end code faster/shorter
def compare_qualitative():
    raise NotImplementedError()


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    train_loader: DataLoader,
    epochs: int,
    device: str = "cpu",
    val_loader: Optional[DataLoader] = None,
    save_best: bool = True,
    verbose: bool = True,
):
    best_state_dict = None
    best_loss = 1e10

    for epoch in trange(epochs, disable=not verbose, leave=False):
        train_running_loss = 0.0
        for x, y, t in tqdm(train_loader, disable=not verbose):
            x, y, t = x.to(device), y.to(device), t.to(device)
            y_pred = model(x) * t
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().cpu().item()
        train_loss = train_running_loss / len(train_loader)

        # TODO: Add flag/hook for tensorboard

        if val_loader is not None:
                val_loss = evaluate(model, criterion, val_loader, device, verbose)

                if save_best and val_loss < best_loss:
                    best_state_dict = model.state_dict()
                    best_loss = val_loss

        if verbose:
            print(f"\nEpoch: {epoch}")
            print(f"\tTraining loss: {train_loss}")
            print(f"\tValidation loss: {val_loss}")

    if save_best and val_loader:
        model.load_state_dict(best_state_dict)

    return model


def to_torchscript(model: nn.Module, file_name: str):
    torch.jit.script(model).save(file_name)