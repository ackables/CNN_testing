"""
CNN Model Structure
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 500, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(500, 1000, kernel_size=5,padding=1, stride=2)
        self.conv3 = nn.Conv2d(1000, 2500, kernel_size=7, padding=1, stride=2)
        # self.conv4 = nn.Conv2d(50, 100, kernel_size=5, stride=2)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(10000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.size())
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout3(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)

        return F.log_softmax(x, dim=1)
    
