# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader


class MNIST(Dataset):

  data_dir = 'MNIST'
  c, h, w = 1, 28, 28
  n_class = 10

  def __init__(self, phase):
    if phase == 'train':
      self.data = open(os.path.join(self.data_dir, 'mnist_train.csv'), 'r').read().split('\n')[:-1]
    elif phase == 'val':
      self.data = open(os.path.join(self.data_dir, 'mnist_test.csv'), 'r').read().split('\n')[:-1]
    self.data = [d.split(',') for d in self.data]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    cls   = int(self.data[idx][0])
    label = np.zeros(self.n_class)
    label[cls] = 1
    label = torch.from_numpy(label).float()
    image = np.array(self.data[idx][1:]).reshape(self.c, self.h, self.w).astype(np.float32)
    image = torch.from_numpy(image).float()
    return {
      'X': image,
      'Y': label
    }


class FashionMNIST(Dataset):

  data_dir = 'fashionMNIST'
  c, h, w = 1, 28, 28
  n_class = 10

  def __init__(self, phase):
    if phase == 'train':
      self.data = open(os.path.join(self.data_dir, 'fashion-mnist_train.csv'), 'r').read().split('\n')[1:-1]
    elif phase == 'val':
      self.data = open(os.path.join(self.data_dir, 'fashion-mnist_test.csv'), 'r').read().split('\n')[1:-1]
    self.data = [d.split(',') for d in self.data]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    cls   = int(self.data[idx][0])
    label = np.zeros(self.n_class)
    label[cls] = 1
    label = torch.from_numpy(label).float()
    image = np.array(self.data[idx][1:]).reshape(self.c, self.h, self.w).astype(np.float32)
    image = torch.from_numpy(image).float()
    return {
      'X': image,
      'Y': label
    }


class MNISTplusFashion(Dataset):

  data_dirM = 'MNIST'
  data_dirF = 'fashionMNIST'
  c, h, w = 1, 28, 28
  n_class = 20

  def __init__(self, phase):
    if phase == 'train':
      self.dataM = open(os.path.join(self.data_dirM, 'mnist_train.csv'), 'r').read().split('\n')[:-1]
      self.dataF = open(os.path.join(self.data_dirF, 'fashion-mnist_train.csv'), 'r').read().split('\n')[1:-1]
    elif phase == 'val':
      self.dataM = open(os.path.join(self.data_dirM, 'mnist_test.csv'), 'r').read().split('\n')[:-1]
      self.dataF = open(os.path.join(self.data_dirF, 'fashion-mnist_test.csv'), 'r').read().split('\n')[1:-1]
    self.dataM = [d.split(',') for d in self.dataM]
    self.dataF = [d.split(',') for d in self.dataF]
    for idx in range(len(self.dataF)):
      self.dataF[idx][0] = int(self.dataF[idx][0]) + 10
    self.data = self.dataM + self.dataF

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    cls   = int(self.data[idx][0])
    label = np.zeros(self.n_class)
    label[cls] = 1
    label = torch.from_numpy(label).float()
    image = np.array(self.data[idx][1:]).reshape(self.c, self.h, self.w).astype(np.float32)
    image = torch.from_numpy(image).float()
    return {
      'X': image,
      'Y': label
    }


if __name__ == "__main__":
  train_data = MNISTplusFashion(phase='train')

  # test a batch
  batch_size = 4
  for i in range(batch_size):
    sample = train_data[i]
    print("sample %d," % i, sample['X'].shape, sample['Y'].shape)

  # test dataloader
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
  for idx, batch in enumerate(train_loader):
    print("{} batch, x.size() {}, y.size() {}".format(idx, batch['X'].size(), batch['Y'].size()))
    assert len(batch['X'].size()) == 4
    assert batch['X'].size()[0] == batch_size
    assert batch['Y'].size()[0] == batch_size
    if idx == 3:
      break
  print("Pass Test")
