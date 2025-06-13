"""
Benjamin Smith
San Francisco State University
College of Science and Engineering
06 June 2025

CNN built to train on the CIFAR-10 dataset
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from Data.loader import cifar10_loader
from Data.testing import test
from Data.training import train
from Model.CNN import Model
import matplotlib.pyplot as plt
import datetime


# set desired device to run model on
device = torch.device("cuda")
epochs = 5
learning_rate = 0.001
log_interval = 100
train_loss = []

def main():
    # load data
    training_loader, test_loader = cifar10_loader()

    # load  model 
    model = Model().to(device)

    # set optimizer function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(1, epochs + 1):
        train_loss.append(train(log_interval, model, device, training_loader, optimizer, epoch))
        test(model, device, test_loader)
        scheduler.step()

    plt.plot([i for i in range(len(train_loss))], train_loss)
    plt.savefig(f"./figures/loss_graph_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")


if __name__ == '__main__':
    main()