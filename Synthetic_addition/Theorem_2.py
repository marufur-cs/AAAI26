import numpy as np
import types
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from decimal_addition import make_dataset, norm, train, evaluate, compute_support_overlap, KAN

x1, s1 = make_dataset(1)
x2, s2 = make_dataset(2)
x3, s3 = make_dataset(3)
x4, s4 = make_dataset(4)
x5, s5 = make_dataset(5)

x1=norm(x1)
x2=norm(x2)
x3=norm(x3)
x4=norm(x4)
x5=norm(x5)

s1=norm(s1)
s2=norm(s2)
s3=norm(s3)
s4=norm(s4)
s5=norm(s5)

tasks = [(x1, s1), (x2, s2), (x3, s3), (x4, s4), (x5, s5)]

# Cumulative Forgetting
Task = 4 # 1 to 4
ex = 5
grid = 20
epochs = 10
forgeting = []
ratio = []
overlap = []
# torch.manual_seed(42)
for _ in range(ex):
  s = [] # Initialize s as an empty list
  model = KAN([2,3,2], grid_size = grid)
  optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
  criterion = nn.MSELoss()

  taskLoss1 = train(model, epochs, optimizer, criterion, Task)
  xx1 = tasks[Task-1][0]
  for nextTask in range(Task+1,6):
    xx2 = tasks[nextTask-1][0]
    s.extend(compute_support_overlap(model, xx1, xx2, t = 0.0)) # Extend the list s

  for nextTask in range(Task+1,6):
    taskLoss2 = train(model, epochs, optimizer, criterion, nextTask)

  prevTaskLoss = evaluate(model, epochs, optimizer, criterion, Task)
  f = prevTaskLoss - taskLoss1
  forgeting.append(f)
  ratio.append(np.round(f/np.sum(s), 2)) # Use np.sum(s)
  overlap.append(np.sum(s)) # Use np.sum(s)

print("Overlap: ",np.mean(overlap))
print("Forgeting: ",np.mean(forgeting))
print("***********************************")