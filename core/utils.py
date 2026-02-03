import torch
import random
import numpy as np
from pathlib import Path


def make_loss_fn(criterion):
    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        loss = criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        return loss, {"acc": acc}
    return loss_fn


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)

    return batch


def save_checkpoint(path, model, optimizer=None, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    return state


def get_dataset_info(loader): 
    x, y = next(iter(loader))

    input_shape = tuple(x.shape[1:])

    if hasattr(loader.dataset, 'classes'):
        num_classes = len(loader.dataset.classes)
    else:
        num_classes = len(torch.unique(y))

    return input_shape, num_classes


def flatten_dim(shape):
    out = 1
    for s in shape:
        out *= s
    return out
