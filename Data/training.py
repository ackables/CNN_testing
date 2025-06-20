"""
Training function
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def train(log_interval, model, device, training_loader, optimizer, epoch, train_loss_list):
    train_loss = train_loss_list
    model.train()
    for batch_idx, (data, target) in enumerate(training_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(training_loader.dataset),
                100. * batch_idx / len(training_loader), loss.item()))
        
    return train_loss