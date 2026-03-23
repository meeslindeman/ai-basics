import torch
import torch.nn as nn
import argparse

from core.utils import seed_everything, get_device, get_dataset_info
from datasets import get_loaders
from models.basic.model import LinearClassifer, MLPClassifier, ResMLP


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)

            output = model(data)
            eval_loss += loss_fn(output, target).item()
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)

    return {"loss": eval_loss / len(dataloader), "accuracy": correct / total}


def main(args):
    seed_everything(args.seed)
    device = get_device()

    train_loader, test_loader = get_loaders(args.dataset, batch_size=args.batch_size)

    input_shape, num_classes = get_dataset_info(train_loader)
    input_size = 1
    for s in input_shape:
        input_size *= s

    if args.model == 'linear':
        model = LinearClassifer(input_size=input_size, num_classes=num_classes).to(device)
    elif args.model == 'mlp':
        model = MLPClassifier(input_size=input_size, num_classes=num_classes, hidden_size=args.hidden_size, dropout=args.dropout).to(device)
    elif args.model == 'resmlp':
        model = ResMLP(input_size=input_size, num_classes=num_classes, hidden_size=args.hidden_size, dropout=args.dropout).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print(f"Training {args.model.upper()} on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_dict = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        eval_dict = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch [{epoch}/{args.epochs}] | Train: loss={train_dict['loss']:.4f} acc={train_dict['accuracy']*100:.2f}% | Test: loss={eval_dict['loss']:.4f} acc={eval_dict['accuracy']*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a basic model on MNIST')
    parser.add_argument('--model', type=str, default='mlp', choices=['linear', 'mlp', 'resmlp'])
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)

    args = parser.parse_args()
    main(args)