import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from core.utils import seed_everything, get_device
from core.loop import train_one_epoch, evaluate, log_epoch
from models.basic.model import LinearClassifer, MLPClassifier, ResMLP
from models.basic.data import get_data_loaders


def make_loss_fn(criterion):
    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        loss = criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        return loss, {"acc": acc}
    return loss_fn


def main(args):
    seed_everything(args.seed)
    device = get_device()

    if args.model == 'linear':
        model = LinearClassifer().to(device)
    elif args.model == 'mlp':
        model = MLPClassifier().to(device)
    elif args.model == 'resmlp':
        model = ResMLP().to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)

    loss_fn = make_loss_fn(nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(1, args.epochs + 1):
        train_dict = train_one_epoch(model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
        eval_dict = evaluate(model, dataloader=test_loader, loss_fn=loss_fn, device=device)

        log_epoch(epoch, args.epochs, train_dict, eval_dict, split_name="Test")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a basic model on MNIST')
    parser.add_argument('--model', type=str, default='mlp', choices=['linear', 'mlp', 'resmlp'], help='Type of model to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay')

    args = parser.parse_args()
    main(args)
