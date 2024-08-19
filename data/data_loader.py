# Handles data loading and preprocessing.
  
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

def get_transforms():
    # Define transforms for training and testing
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    return train_transform, test_transform

def get_loaders(config):
    train_transform, test_transform = get_transforms()
    
    # Load the CIFAR10 dataset
    train_dataset = CIFAR10(root=config['dataset_path'], train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root=config['dataset_path'], train=False, download=True, transform=test_transform)

    # Creating data indices for training and validation splits:
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(config['validation_split'] * num_train))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'])
    val_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=val_sampler, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader, test_loader
