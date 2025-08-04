import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import torch.nn.init as init

class   MLPLinear(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        
        self.linear = linear_layer
        
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.rank = rank
        self.alpha = alpha
        
        self.lora_a = nn.Linear(self.in_features, self.rank)
        self.lora_b = nn.Linear(self.rank, self.out_features)
        
    def forward(self, x):
        x1 = self.linear(x)
        x2 = (self.alpha/self.rank) * self.lora_b(self.lora_a(x))
        out = x1 + x2
        return out

class KANHead(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()

        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha

        self.in_features = self.linear.in_features

        self.lora_a = nn.Linear(self.in_features, self.rank)
        self.lora_b = nn.Linear(self.rank, self.in_features)

    def forward(self, x):

        x = (self.alpha/self.rank) * self.lora_b(self.lora_a(x))
        x = self.linear(x)

        return x