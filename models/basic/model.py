import torch.nn as nn
import torch.nn.functional as F

class LinearClassifer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifer, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        out = self.fc(x)
        return out
    
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)  
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out
    
class ResMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, dropout=0.5):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        residual = F.relu(self.fc1(x))
        out = F.relu(self.fc2(residual))
        out += residual  
        out = self.dropout(out)
        out = self.fc3(out)
        return out