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
        for x, y in tqdm(loader, disable=not verbose, leave=False):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            running_loss += criterion(y_pred, y).cpu().item()

    return running_loss / len(loader)


# TODO: We might want to include some of this functionality within the nn.Module itself (i.e. subclass it)
# This way the torchscripted module would also contain that functionality and this might make end code faster/shorter
def compare_qualitative(model: nn.Module, xx: torch.Tensor, yy: torch.Tensor, start_idx: int, n_steps: int, delay_steps: int):
        one_steps = model(xx[:-1,:])
        start_state = yy[start_idx,:3]
        ossi = start_idx - delay_steps
        relevant_one_steps = one_steps[ossi:ossi+n_steps]
        seq = rollout_single_sequence(relevant_one_steps, start_state)
        tx = yy[:,0]
        ty = yy[:,1]
        mx = seq[:,0]
        my = seq[:,1]
        return tx, ty, mx, my


def rollout_single_sequence(one_steps: torch.Tensor, start_state: torch.Tensor):
    n_steps = one_steps.shape[0]
    seq = torch.zeros((n_steps+1, one_steps.shape[1]))
    seq[0] = start_state
    for t in range(n_steps):
        curr_angle = seq[t,2]
        summand = one_steps[t,:]
        assert(summand.shape == (3,))
        world_summand = torch.zeros_like(summand)
        world_summand[0] = torch.cos(curr_angle) * summand[0] - torch.sin(curr_angle) * summand[1]
        world_summand[1] = torch.sin(curr_angle) * summand[0] + torch.cos(curr_angle) * summand[1]
        world_summand[2] = summand[2]
        seq[t+1,:] = seq[t,:] + world_summand
    return seq


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
        for x, y in tqdm(train_loader, disable=not verbose):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
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