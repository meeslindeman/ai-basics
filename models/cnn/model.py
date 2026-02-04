import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, input_shape, base_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(base_channels * 4, num_classes)

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
    

class SmallResNet(nn.Module):
    def __init__(self, input_shape, num_classes, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        pass