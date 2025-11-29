import numpy as np
import types
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class Quantize:
    def __init__(self, n_levels):
        self.n_levels = n_levels

    def __call__(self, tensor):
        # Assumes tensor is in [0, 1], scale to [0, 255] if needed
        tensor = (tensor * (self.n_levels - 1)).round() / (self.n_levels - 1)
        return tensor

def get_task_data(dataset, classes):
    indices = []
    indices1 = [i for i, (_, label) in enumerate(dataset) if label == classes[0]]
    indices2 = [i for i, (_, label) in enumerate(dataset) if label == classes[1]]
    for idx1, idx2 in zip(indices1, indices2):
        indices.append(idx1)
        indices.append(idx2)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=2, shuffle=False)

def get_tasks(dataset_name, resize = None, quantize = None):
    # Define the transformation for the dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    quantizedTransform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize, resize)) if resize else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        Quantize(quantize),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if quantize:
            print("Using quantized dataset")
            transform = quantizedTransform
            
    if dataset_name == "cifar10": # Shape is 32 for CIFAR-10
        print("Using CIFAR-10 dataset")
        train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)

        task1 = get_task_data(train_dataset, (1, 2))
        task2 = get_task_data(train_dataset, (3, 4))
        task3 = get_task_data(train_dataset, (5, 6))
        task4 = get_task_data(train_dataset, (7, 8))
        task5 = get_task_data(train_dataset, (9, 0))
        return task1, task2, task3, task4, task5
    
    elif dataset_name == "tiny_imagenet": # Shape is 64 for Tiny ImageNet
        data_dir = '/deac/csc/yangGrp/rahmm224/datasets/tiny-imagenet-200'
        train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)

        task1 = get_task_data(train_dataset, (0, 1))
        task2 = get_task_data(train_dataset, (2, 3))
        task3 = get_task_data(train_dataset, (4, 5))
        task4 = get_task_data(train_dataset, (6, 7))
        task5 = get_task_data(train_dataset, (8, 9))
        return task1, task2, task3, task4, task5
    
    elif dataset_name == "mnist": # Shape is 28 for MNIST
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

        task1 = get_task_data(train_dataset, (1, 2))
        task2 = get_task_data(train_dataset, (3, 4))
        task3 = get_task_data(train_dataset, (5, 6))
        task4 = get_task_data(train_dataset, (7, 8))
        task5 = get_task_data(train_dataset, (9, 0))
        return task1, task2, task3, task4, task5