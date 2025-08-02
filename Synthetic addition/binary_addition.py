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

def to_binary(n):
  if n > 15:
    print(f'Only 4 bit numbers')
    return

  arr = np.zeros(5)
  b = bin(n)[2:]
  l = len(b)
  for j in range(l):
    arr[j] = int(b[l - j - 1])
  return arr

def make_dataset(t):
  x1 = to_binary(t)
  for i in range(9):
    x1 = np.vstack((x1,to_binary(t)))

  y1 = np.zeros(5)
  for i in range(1, 10):
    y1 = np.vstack((y1,to_binary(i)))

  x1 = np.vstack((x1,y1))
  y1 = np.vstack((y1,x1[:10]))

  s1 = []
  x = []
  for i in range(20):
    t = []
    s = []
    c = 0
    for j in range(5):
      t.append([x1[i][j], y1[i][j]])
      a = (x1[i][j] + y1[i][j] + c)
      s.append([a, 0] if a<2 else [a%2, 1.])
      c = 1 if a>1 else 0
    x.append(t)
    s1.append(s)
  x = np.array(x)
  s1 = np.array(s1)

  return x, s1

x1, s1 = make_dataset(1) #[x4, y4], [x3, y3],...,[x0, y0]
x2, s2 = make_dataset(2)
x3, s3 = make_dataset(3)
x4, s4 = make_dataset(4)
x5, s5 = make_dataset(5)

import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def get_activation_support(self, input_samples: torch.Tensor):
      """Return the activation mask for each (p, q) in the spline."""
      # x.shape: [N, in_features]
      bases = self.b_splines(input_samples)  # shape: (N, in_features, grid_size + spline_order)
      active_mask = bases.abs() > 1e-2       # binary mask of activity
      return active_mask  # shape: (N, in_features, #basis)


    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        # Support
        self.task_support = []
        self.cal_support = False
        # Support

    def forward(self, x: torch.Tensor, update_grid=False):
      c = torch.tensor(0.0, dtype=x.dtype)
      res = []
      for i in range(5):
          a = torch.tensor([x[i, 0], x[i, 1], c])

          for l, layer in enumerate(self.layers):
              if update_grid:
                  layer.update_grid(a)

              # Support
              if self.cal_support:
                if len(self.task_support) != len(self.layers):
                  self.task_support.append(layer.get_activation_support(a.reshape(1,-1)))
                else:
                  sup1 = self.task_support[l]
                  sup2 = layer.get_activation_support(a.reshape(1,-1))
                  sup = sup1 | sup2
                  self.task_support[l] = sup

              # Support

              a = layer(a)
              if self.training == False:
                print(a)

          c = a[1]
          res.append(torch.stack([a[0], c]))

      return torch.stack(res)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

def compute_support_overlap(support_i, support_j):
    # both are tensors: [N, in_features, basis]
    intersection = (support_i & support_j).sum(dim=0)  # sum over samples
    union = (support_i | support_j).sum(dim=0)
    overlap = (intersection.float() / union.float().clamp(min=1e-6)).max().item()
    return overlap

model = KAN([3,2,2], grid_size = 5)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = nn.MSELoss()
epochs = 10

# Trining on Task 1 (Ones addition)
loss_task1 = []
loss_task2 = []
loss_task3 = []
loss_task4 = []
loss_task5 = []
for epoch in tqdm(range(epochs)):
    # Train
    model.train()
    if epoch == epochs-1:
      model.cal_support = True
    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      optimizer.zero_grad()
      pred = model(input).squeeze()
      loss = criterion(pred, result)
      loss.backward()
      optimizer.step()
      loss_temp += loss.item()

    loss_task1.append(loss_temp/len(x1))

    model.cal_support = False

    loss_temp = 0.0
    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task2.append(loss_temp/len(x2))

    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task3.append(loss_temp/len(x3))

    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task5.append(loss_temp/len(x5))

task1_support = model.task_support

# Trining on Task 2 (Twos addition)
for epoch in tqdm(range(epochs)):
    # Train
    model.train()
    loss_temp = 0.0

    # Support
    model.cal_support = False
    if epoch == epochs-1:
      model.cal_support = True
      model.task_support = []
    # Support

    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      optimizer.zero_grad()
      pred = model(input).squeeze()
      loss = criterion(pred, result)
      loss.backward()
      optimizer.step()
      loss_temp += loss.item()

    loss_task2.append(loss_temp/len(x2))

    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task1.append(loss_temp/len(x1))

    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task3.append(loss_temp/len(x3))


    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task5.append(loss_temp/len(x5))

task2_support = model.task_support

# Trining on Task 3 (Twos addition)
for epoch in tqdm(range(epochs)):
    # Train
    model.train()
    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      optimizer.zero_grad()
      pred = model(input).squeeze()
      loss = criterion(pred, result)
      loss.backward()
      optimizer.step()
      loss_temp += loss.item()

    loss_task3.append(loss_temp/len(x3))

    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task1.append(loss_temp/len(x1))

    loss_temp = 0.0
    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task2.append(loss_temp/len(x2))


    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task5.append(loss_temp/len(x5))

# Trining on Task 4 (Twos addition)
for epoch in tqdm(range(epochs)):
    # Train
    model.train()
    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      optimizer.zero_grad()
      pred = model(input).squeeze()
      loss = criterion(pred, result)
      loss.backward()
      optimizer.step()
      loss_temp += loss.item()

    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task1.append(loss_temp/len(x1))

    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task3.append(loss_temp/len(x3))


    loss_temp = 0.0
    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task2.append(loss_temp/len(x2))

    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task5.append(loss_temp/len(x5))

# Trining on Task 5 (Twos addition)
for epoch in tqdm(range(epochs)):
    # Train
    model.train()
    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      optimizer.zero_grad()
      pred = model(input).squeeze()
      loss = criterion(pred, result)
      loss.backward()
      optimizer.step()
      loss_temp += loss.item()

    loss_task5.append(loss_temp/len(x5))

    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task1.append(loss_temp/len(x1))

    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task3.append(loss_temp/len(x3))


    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      pred = (model(input).squeeze()>0.5).int()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task2.append(loss_temp/len(x2))