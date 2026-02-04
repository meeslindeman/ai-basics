import torch
import torch.nn as nn
import argparse

from core.utils import seed_everything, get_device, get_dataset_info, make_loss_fn
from core.loop import train_one_epoch, evaluate, log_epoch
from datasets import get_loaders
from models.cnn.model import CNNClassifier, SmallResNet

def main(args):
    seed_everything(args.seed)
    device = get_device()

    train_loader, test_loader = get_loaders(args.dataset, batch_size=args.batch_size)

    input_shape, num_classes = get_dataset_info(train_loader)

    if args.model == 'cnn':
        model = CNNClassifier(input_shape=input_shape, base_channels=args.base_channels, num_classes=num_classes).to(device)
    elif args.model == 'resnet':
        model = SmallResNet(input_shape=input_shape, num_classes=num_classes, dropout=args.dropout).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    loss_fn = make_loss_fn(nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(1, args.epochs + 1):
        train_dict = train_one_epoch(model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
        eval_dict = evaluate(model, dataloader=test_loader, loss_fn=loss_fn, device=device)

        log_epoch(epoch, args.epochs, train_dict, eval_dict, split_name="Test")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a basic model on MNIST')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'], help='Type of model to use')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size for CNN/ResNet')
    parser.add_argument('--base-channels', type=int, default=32, help='Base number of channels for CNN')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

    args = parser.parse_args()
    main(args)
