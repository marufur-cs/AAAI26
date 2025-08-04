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


def norm(x):
  min  = 0
  max = 10
  return (x - min) / (max - min)

def make_dataset(t):
  x = []
  s = []
  b = 10
  for i in range(b):
    sum = t + i
    if sum > 9:
      # continue
      x.append([t, i])
      s.append([sum-10., 1.])
      x.append([i, t])
      s.append([sum-10., 1.])
    else:
      x.append([t, i])
      s.append([sum, 0.])
      x.append([i, t])
      s.append([sum, 0.])

  x = np.array(x)
  s = np.array(s)
  return x, s

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

    def get_activation_support(self, input_samples: torch.Tensor, t):
      """Return the activation mask for each (p, q) in the spline."""
      # x.shape: [N, in_features]
      bases = self.b_splines(input_samples)  # shape: (N, in_features, grid_size + spline_order)
      active_mask = (bases.abs() > t)    # binary mask of activity
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

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

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

    def forward(self, x: torch.Tensor, update_grid=False):
      for l, layer in enumerate(self.layers):
          if update_grid:
              layer.update_grid(x)
          x = layer(x)
      return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

def get_task_support(model, task, t):
      task_support = []
      model.eval()
      with torch.no_grad():
        for i in range(len(task)):
          x = torch.from_numpy(task[i]).float()
          for l, layer in enumerate(model.layers):
              if len(task_support) != len(model.layers):
                task_support.append(layer.get_activation_support(x.reshape(1,-1), t))
              else:
                sup1 = task_support[l]
                sup2 = layer.get_activation_support(x.reshape(1,-1), t)
                sup = sup1 | sup2
                task_support[l] = sup
              x = layer(x)
      return task_support

def compute_support_overlap(model, task1, task2, t):
    support_1 = get_task_support(model, task1, t)
    support_2 = get_task_support(model, task2, t)

    overlap = []
    for sup1, sup2 in zip(support_1, support_2):
      # both are tensors: [N, in_features, basis]
      overlap_ = (sup1 & sup2).sum()
      overlap_layer = (overlap_ / sup1.numel()).item()
      overlap.append(overlap_layer)
    return overlap

def compute_support_union(model, task1, task2, t):
    support_1 = get_task_support(model, task1, t)
    support_2 = get_task_support(model, task2, t)
    union = []
    for sup1, sup2 in zip(support_1, support_2):
      union.append(sup1 | sup2)
    return union


def train(model, epochs, optimizer, criterion, T):
  (trainx, trains) = tasks[T-1]
  if T > 1:
    (evalx, evals) = tasks[T-2]
  for epoch in tqdm(range(epochs)):
      # Train
      model.train()
      loss_temp = 0.0
      for i in range(len(trainx)):
        input = torch.from_numpy(trainx[i]).float()
        result = torch.from_numpy(trains[i]).float()

        optimizer.zero_grad()
        pred = model(input).squeeze()
        loss = criterion(pred, result)
        loss.backward()
        optimizer.step()
        loss_temp += loss.item()
      losst1 = loss_temp/len(trainx)

  # print(f"Loss on task{T}: {losst1}")
  # print(f"Loss on task{T-1}: {loss_temp/len(x2)}")
  return losst1

def evaluate(model, epochs, optimizer, criterion, T):
  (evalx, evals) = tasks[T-1]
  loss_temp = 0.0
  for i in range(len(evalx)):
    input = torch.from_numpy(evalx[i]).float()
    result = torch.from_numpy(evals[i]).float()

    pred = model(input).squeeze()
    loss = criterion(pred, result)
    loss_temp += loss.item()

  losst2 = loss_temp/len(evalx)
  return losst2


model = KAN([2,3,2], grid_size = 5)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = nn.MSELoss()
epochs = 10

loss_task1 = []
loss_task2 = []
loss_task3 = []
loss_task4 = []
loss_task5 = []
'''
# Trining on Task 1 (Ones addition)
for epoch in tqdm(range(epochs)):
    # Train
    model.train()

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
    losst1 = loss_temp/len(x1)

    loss_temp = 0.0
    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task2.append(loss_temp/len(x2))

    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task3.append(loss_temp/len(x3))

    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task5.append(loss_temp/len(x5))

# Trining on Task 2 (Twos addition)
for epoch in tqdm(range(epochs)):
    # Train
    model.train()

    loss_temp = 0.0
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
    losst2 = loss_temp/len(x2)

    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task1.append(loss_temp/len(x1))
    losst1 = loss_temp/len(x1)

    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task3.append(loss_temp/len(x3))


    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task5.append(loss_temp/len(x5))

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
    losst3 = loss_temp/len(x3)

    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task1.append(loss_temp/len(x1))

    loss_temp = 0.0
    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task2.append(loss_temp/len(x2))
    losst2 = loss_temp/len(x2)

    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      pred = model(input).squeeze()
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
    losst4 = loss_temp/len(x4)

    loss_task4.append(loss_temp/len(x4))

    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task1.append(loss_temp/len(x1))

    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task3.append(loss_temp/len(x3))
    losst3 = loss_temp/len(x3)

    loss_temp = 0.0
    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task2.append(loss_temp/len(x2))

    loss_temp = 0.0
    for i in range(len(x5)):
      input = torch.from_numpy(x5[i]).float()
      result = torch.from_numpy(s5[i]).float()

      pred = model(input).squeeze()
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
    losst5 = loss_temp/len(x5)

    loss_temp = 0.0
    for i in range(len(x1)):
      input = torch.from_numpy(x1[i]).float()
      result = torch.from_numpy(s1[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task1.append(loss_temp/len(x1))

    loss_temp = 0.0
    for i in range(len(x3)):
      input = torch.from_numpy(x3[i]).float()
      result = torch.from_numpy(s3[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task3.append(loss_temp/len(x3))


    loss_temp = 0.0
    for i in range(len(x4)):
      input = torch.from_numpy(x4[i]).float()
      result = torch.from_numpy(s4[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task4.append(loss_temp/len(x4))
    losst4 = loss_temp/len(x4)

    loss_temp = 0.0
    for i in range(len(x2)):
      input = torch.from_numpy(x2[i]).float()
      result = torch.from_numpy(s2[i]).float()

      pred = model(input).squeeze()
      # print(pred, result)
      loss = criterion(pred, result)
      loss_temp += loss.item()
    loss_task2.append(loss_temp/len(x2))
    '''