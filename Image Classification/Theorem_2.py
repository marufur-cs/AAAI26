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
tasks = [task1c, task2c, task3c, task4c, task5c]
shape = 32 # Shape is 32 for CIFAR-10

# Cumulative Forgetting (Theorem 2)
task_support = {}
ex = 5
grid = 20
epochs = 10
bc = 5
# torch.manual_seed(42)
for Task in tqdm(range(1,5)): # Task 1 to 4
  forget = []
  overlap = []
  ratio = []
  for e in range(ex):

    s = [] # Initialize s as an empty list
    model = KANTransformer(grid_size = grid, layers_hidden = [shape*shape,10], embed_dim=shape, seq_length=shape, num_encoders=1, expansion_factor=1, n_heads=4)

    t1 = tasks[Task-1]
    task1Loss = train(epochs, bc, shape, model, device, t1)

    task1_support = get_task_support(bc, shape, model, device, t1)
    for nextTask in range(Task+1,6):
      t2 = tasks[nextTask-1]
      task2_support = get_task_support(bc, shape, model, device, t2)
      s.extend(compute_support_overlap(task1_support, task2_support))

    for nextTask in range(Task+1,6):
      taskLoss2 = train(epochs, bc, shape, model, device, tasks[nextTask-1])

    prevTask1Loss = eval_(bc, shape, model, device, t1)

    f = prevTask1Loss - task1Loss
    forget.append(f)
    ratio.append(f/np.sum(s)) # Use np.sum(s)
    overlap.append(np.sum(s)) # Use np.sum(s)

  print("Overlap: ",np.mean(overlap))
  print("Forgeting: ",np.mean(forget))
  print("Ratio: ",np.mean(ratio))
  print("***********************************")
