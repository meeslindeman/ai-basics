import torch
import torch.nn as nn
import argparse

from core.utils import seed_everything, get_device
from datasets import get_loaders
from models.rnn.model import CharRNN


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    hidden = None
    total_loss = 0.0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        if hidden is not None:
            hidden = hidden.detach()  
        loss = loss_fn(output.view(-1, output.size(-1)), target.view(-1))
        pred = output.argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += target.numel()
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  
        optimizer.step()

    total_loss /= len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return {"loss": total_loss, "accuracy": accuracy}

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    hidden = None
    eval_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output, hidden = model(data, hidden)
            if hidden is not None:
                hidden = hidden.detach()  
            eval_loss += loss_fn(output.view(-1, output.size(-1)), target.view(-1)).item()
            pred = output.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += target.numel()

    eval_loss /= len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return {"loss": eval_loss, "accuracy": accuracy}

def main(args):
    seed_everything(args.seed)
    device = get_device()

    train_loader, test_loader = get_loaders(
        args.dataset, batch_size=args.batch_size, sequence_length=args.sequence_length
    )

    vocab_size = train_loader.dataset.dataset.vocab_size  

    if args.model == 'rnn':
        model = CharRNN(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(1, args.epochs + 1):
        train_dict = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        eval_dict = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch [{epoch}/{args.epochs}] | Train: loss={train_dict['loss']:.4f} acc={train_dict['accuracy']*100:.2f}% | Test: loss={eval_dict['loss']:.4f} acc={eval_dict['accuracy']*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='shakespeare')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'gru'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--sequence-length', type=int, default=100)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)

    args = parser.parse_args()
    main(args)

