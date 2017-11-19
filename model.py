# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

class Multitask(nn.Module):

  def __init__(self, n_class):
    super().__init__()

    self.n_class = n_class
    self.features = nn.Sequential(
      nn.Conv2d(1, 96, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(96, 96, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(96, 96, 3, padding=1, stride=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(96, 192, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(192, 192, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(192, 192, 3, padding=1, stride=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(192, 192, 3),
      nn.ReLU(inplace=True),
      nn.Conv2d(192, 192, 1),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Conv2d(192, self.n_class, 1)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    avg_pool = nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)))
    x = avg_pool(x).view(-1, self.n_class)  # shape=(batch_size, n_class)
    return x


if __name__ == '__main__':
  # check output shape
  batch, channel, h, w = 10, 1, 28, 28
  n_class = 10
  model = Multitask(n_class=n_class)
  input = torch.autograd.Variable(torch.randn(batch, channel, h, w))
  output = model(input)
  print(output.size())
  assert len(output.size()) == 2
  assert output.size()[0] == batch
  assert output.size()[1] == n_class
  print("Pass Test")
