import torch
import logging

from .utils import to_device


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


def _mean_dict(logs):
    out = {}
    for log in logs:
        for k, v in log.items():
            out.setdefault(k, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in out.items()}


def train_one_epoch(
        model,
        dataloader,
        optimizer,
        loss_fn,
        device,
        grad_clip: float | None = None
    ):
    model.train()
    logs_list = []

    for step, batch in enumerate(dataloader):
        batch = to_device(batch, device)
        
        loss, logs = loss_fn(model, batch)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        logs_list.append({'loss': loss.item(), **logs})

    return _mean_dict(logs_list)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    logs_list = []
    for batch in dataloader:
        batch = to_device(batch, device)
        
        loss, logs = loss_fn(model, batch)
        logs_list.append({'loss': loss.item(), **logs})

    return _mean_dict(logs_list)


def log_epoch(epoch: int, total_epochs: int, train: dict, val: dict | None = None, split_name: str = "Test"):
    def fmt(d: dict):
        parts = []
        for k, v in d.items():
            if "acc" in k.lower():
                parts.append(f"{k}={v*100:.2f}%")
            else:
                parts.append(f"{k}={v:.4f}")
        return " ".join(parts)

    msg = f"Epoch [{epoch}/{total_epochs}] | Train: {fmt(train)}"

    if val is not None:
        msg += f" | {split_name}: {fmt(val)}"

    logger.info(msg)




