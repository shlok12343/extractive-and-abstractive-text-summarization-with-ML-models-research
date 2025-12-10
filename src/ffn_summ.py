import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# feedforward neural network definition
class FF_Net_Summary(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 1) # single output

    def forward(self, x):
        f1 = F.relu(self.fc1(x))
        d1 = self.drop1(f1)

        f2 = F.relu(self.fc2(d1))
        d2 = self.drop2(f2)

        output = self.fc3(d2)
        return output