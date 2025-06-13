"""
Loads the CIFAR-10 dataset
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def cifar10_loader(data_dir="./CIFAR-10"):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    training_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    batch_size = 32


    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    return training_loader, test_loader