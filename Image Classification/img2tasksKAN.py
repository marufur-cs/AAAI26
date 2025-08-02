import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from load_data import *
from code_imageEx.KANTransformer import *
import csv
import argparse

parser = argparse.ArgumentParser(description="KAN Transformer Training")
parser.add_argument('--dataset', type=str, help="dataset name")
args = parser.parse_args()

if args.dataset == 'cifar10':
    shape = 32
elif args.dataset == 'tiny-imagenet':
    shape = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10 dataset
task1, task2, task3, task4,task5 = get_tasks(args.dataset)

model_config = {1:(5,1,1,1),
                2:(10,1,1,1),
                3:(10,1,1,2),
                4:(10,1,1,4),
                5:(10,1,2,1),
                6:(10,1,2,2),
                7:(10,1,5,1),
                8:(10,1,5,2),
                9:(10,1,5,4),
                10:(10,2,1,4)}


epochs = 10
exp = 5
for m in tqdm(range(1,11)):
  grid_size,cl,nb,h = model_config[m]
  print(grid_size,cl,nb,h)
  if cl==1:
      layers=[shape*shape,10]
  else:
      layers=[shape*shape,100,10]

  for e in tqdm(range(exp)):
    acc = []
    for bc in tqdm(range(1, 11)):
      model = KANTransformer(grid_size,layers, embed_dim=shape, seq_length=shape, num_encoders=nb, expansion_factor=1, n_heads=h)

      train(epochs, bc, shape, model, device, task1)
      train(epochs, bc, shape, model, device, task2)

      a = evaluate(bc, shape, model, device, task1)
      acc.append(a)
    with open('dataFigure4.csv', 'a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(acc)
