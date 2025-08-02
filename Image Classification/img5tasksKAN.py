import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from load_data import *
from KANTransformer import *
import csv
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10 dataset
task1c, task2c, task3c, task4c,task5c = get_tasks("cifar10")
cifartasks = [task1c, task2c, task3c, task4c, task5c]
task1t, task2t, task3t, task4t,task5t = get_tasks("tiny_imagenet")
tinytasks = [task1t, task2t, task3t, task4t, task5t]

epochs = 10
exp = 5

grid_size,cl,nb,h = (10,1,1,1)
print(grid_size,cl,nb,h)

for t in tqdm(range(2, 6)):
    for e in tqdm(range(exp)):
        acc = []
        for bc in tqdm(range(1, 11)):
            a = 0.0
            shape = 32 # Shape is 32 for CIFAR-10
            layers=[shape*shape,10]
            model = KANTransformer(grid_size,layers, embed_dim=shape, seq_length=shape, num_encoders=nb, expansion_factor=1, n_heads=h)
            for i in range(t):
                train(epochs, bc, shape, model, device, cifartasks[i])
            for i in range(t-1):
                a += evaluate(bc, shape, model, device, cifartasks[i])
            acc.append(a/(t-1))
        with open('dataFigure5.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(acc)
            
    for e in tqdm(range(exp)):   
        acc = []
        for bc in tqdm(range(1, 11)):
            a = 0.0
            shape = 64 # Shape is 64 for Tiny ImageNet
            layers=[shape*shape,10]
            model = KANTransformer(grid_size,layers, embed_dim=shape, seq_length=shape, num_encoders=nb, expansion_factor=1, n_heads=h)
            for i in range(t):
                train(epochs, bc, shape, model, device, tinytasks[i])
            for i in range(t-1):
                a += evaluate(bc, shape, model, device, tinytasks[i])
            acc.append(a/(t-1))
            
        with open('dataFigure5.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(acc)
