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
epochs = 15
learning_rate = 0.0001
log_interval = 100
batch_size = 16

def main():
    train_loss = []
    test_loss = []
    accuracy = []
    # load data
    training_loader, test_loader = cifar10_loader(batch_size=batch_size)

    # load  model 
    model = Model().to(device)

    # set optimizer function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=3, gamma=0.9)
    for epoch in range(1, epochs + 1):
        train_loss = (train(log_interval, model, device, training_loader, optimizer, epoch, train_loss_list=train_loss))
        test_loss, accuracy = (test(model, device, test_loader, test_loss_list=test_loss, accuracy_list=accuracy))
        scheduler.step()

    plt.plot([i/(50000//batch_size) for i in range(len(train_loss))], train_loss)
    # plt.savefig(f"./figures/loss_graph_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")

    plt.plot([i/(10000//batch_size) for i in range(len(test_loss))], test_loss)
    plt.plot([i for i in range(len(accuracy))], accuracy)
    plt.savefig(f"./figures/loss_graph_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")


if __name__ == '__main__':
    main()