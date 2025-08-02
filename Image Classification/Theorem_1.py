import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from load_data import *
from KANTransformer import *
import csv
import argparse
# Theorem 1: Retention Bound
    # Theorem 3: Intrinsic Dimention Forgetting Rate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = "cifar10"  # Change to "tiny_imagenet" or "mnist" as needed
shape = 32  # Set shape based on dataset (32 for CIFAR-10, 64 for Tiny ImageNet, 28 for MNIST)
# Load CIFAR-10 dataset
task1c, task2c, task3c, task4c, task5c = get_tasks(dataset)
tasks = [task1c, task2c, task3c, task4c, task5c]

allPossibleF = [] 
for grid in [10, 15, 20]:  # Different grid sizes   
    for T in tqdm(range(1,5)): 
        bc = 5
        ex = 5
        # grid = 20
        epochs = 10
        
        forget = []
        overlap = []
        ratio = []
        for e in range(ex):
            model = KANTransformer(grid_size = grid, layers_hidden = [shape*shape, 10], embed_dim=shape, seq_length=shape, num_encoders=1, expansion_factor=1, n_heads=1)

            t1 = tasks[T-1]
            t2 = tasks[T]

            task1Loss = train(epochs, bc, shape, model, device, t1)

            task1_support = get_task_support(bc, shape, model, device, t1)
            task2_support = get_task_support(bc, shape, model, device, t2)

            s = np.max(compute_support_overlap(task1_support, task2_support))

            task2Loss = train(epochs, bc, shape, model, device, t2)

            prevTask1Loss = eval_(bc, shape, model, device, t1)

            f = prevTask1Loss - task1Loss
            forget.append(f)
            ratio.append(f/s)
            overlap.append(s)
            allPossibleF.append(f)
        print(f"Grid: {grid}, Task: {T}")
        print("Forgeting: ",np.mean(forget))
        print("Overlap: ",np.mean(overlap))
        print("Ratio: ",np.mean(ratio))
        # print("***********************************")
print(allPossibleF)