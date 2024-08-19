# This module could contain any additional data transformations or utility functions.

from torchvision import transforms

def basic_transforms():
    # Standard transforms for the training set
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flips the image horizontally
        transforms.RandomCrop(32, padding=4),  # Pads the image by 4 pixels on each side and randomly crops a patch of 32x32
        transforms.ToTensor(),  # Converts the image to a PyTorch tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Normalizes the tensor with mean and std for CIFAR-10
    ])
    
    # Standard transforms for the test set (no augmentation)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),  # Converts the image to a PyTorch tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Normalizes the tensor with mean and std for CIFAR-10
    ])

    return train_transforms, test_transforms
