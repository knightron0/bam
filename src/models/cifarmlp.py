import torch
import torch.nn as nn

class CifarMLP(nn.Module):
    def __init__(self):
        super(CifarMLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256, bias=False)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.fc3 = nn.Linear(128, 32, bias=False)
        self.fc4 = nn.Linear(32, 10, bias=False)

    def forward(self, x):
        x = x.reshape(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
