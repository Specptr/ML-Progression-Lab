import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(batch_size=64):
    """
    Load MNIST dataset
    create DataLoader objects
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081, ))
    ])
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader


def plot_images(images, labels, preds=None):
    """
    Display a grid of images with true labels and optional predictions.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(12,3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap='gray')
        title = f"Label: {labels[i].item()}"
        if preds is not None:
            title += f"\nPred: {preds[i].item()}"
        ax.set_title(title)
        ax.axis('off')
    plt.show()


def plot_loss_accuracy(loss_list, acc_list):
    """
    Plot loss and accuracy curves over epochs
    """
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(loss_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(acc_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Loss and Accuracy")
    plt.show()