import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """
    A simple CNN-based classifier for image classification tasks.
    """
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(out_channels * 4, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out
    

class ResidualBlock(nn.Module):
    """
    A simple residual block for building a small ResNet architecture.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.projection is not None:
            identity = self.projection(identity)
        x += identity
        x = F.relu(x)
        return x
    

class SmallResNet(nn.Module):
    """
    A small ResNet architecture for image classification tasks, using residual blocks.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, 64)
        self.block2 = ResidualBlock(64, 128, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out