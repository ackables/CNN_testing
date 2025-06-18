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
        self.conv1 = nn.Conv2d(3, 1000, kernel_size=5, padding=(5-2)//2, stride=2)
        self.batchNorm1 = nn.BatchNorm2d(1000)
        self.conv2 = nn.Conv2d(1000, 2000, kernel_size=3, padding='same', stride=1)
        self.batchNorm2 = nn.BatchNorm2d(2000)
        self.conv3 = nn.Conv2d(2000, 5000, kernel_size=3, padding='same', stride=1)
        self.batchNorm3 = nn.BatchNorm2d(5000)
        self.conv4 = nn.Conv2d(5000, 512, kernel_size=1)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.35)
        # self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(4608, 1000)
        self.fc2 = nn.Linear(1000, 10)
        # self.fc3 = nn.Linear(100, 10)

        self.lrn = nn.LocalResponseNorm(3)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = self.batchNorm1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = self.batchNorm2(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = self.batchNorm3(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)

        x = F.max_pool2d(x, kernel_size=4)
        x = self.lrn(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        # x = self.dropout3(x)
        # x = F.leaky_relu(self.fc3(x), negative_slope=0.01)

        # return F.log_softmax(x, dim=1)
        return x
    
