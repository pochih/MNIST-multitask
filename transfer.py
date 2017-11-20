# -*- coding: utf8

from __future__ import print_function

import numpy as np
import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from dataset import MNIST, FashionMNIST
from model import Multitask

parser = argparse.ArgumentParser(description='PyTorch CNN Sentence Classification')
# training configs
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='training optimizer (default: Adam)')
parser.add_argument('--batch-size', type=int, default=100,
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100,
                    help='input batch size for testing (default: 100)')
parser.add_argument('--n-class', type=int, default=2,
                    help='number of class (default: 2)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--w-decay', type=float, default=0.,
                    help='L2 norm (default: 0)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=500,
                    help='how many batches to wait before logging training status')
parser.add_argument('--pre-trained', type=int, default=1,
                    help='using pre-trained model or not (default: 1)')
# data
parser.add_argument('--source-dataset', type=str, default='M+F',
                    help='source dataset')
parser.add_argument('--target-dataset', type=str, default='M',
                    help='current dataset')
# device
parser.add_argument('--cuda', type=int, default=1,
                    help='using CUDA training')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='using multi-gpu')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
params = "{}-{}-batch{}-epoch{}-lr{}-momentum{}-wdecay{}".format(args.source_dataset, args.optimizer, args.batch_size, args.epochs, args.lr, args.momentum, args.w_decay)
print('args: {}\nparams: {}'.format(args, params))

# define result file & model file
result_dir = 'result'
model_dir = 'model'
for dir in [result_dir, model_dir]:
  if not os.path.exists(dir):
    os.makedirs(dir)
accs = np.zeros(args.epochs)

# load data
train_data   = MNIST(phase='train')
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_data     = MNIST(phase='val')
val_loader   = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

# load pre-trained model
if args.pre_trained:
  pretrained_model = torch.load(os.path.join(model_dir, params))
  pretrained_dict = pretrained_model.state_dict()
  del pretrained_dict['classifier.weight']
  del pretrained_dict['classifier.bias']
  print("Load pre-trained model")

# use pre-trained weight
model = Multitask(n_class=train_data.n_class)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # filter out unnecessary keys
model_dict.update(pretrained_dict)  # overwrite entries in the existing state dict
model.load_state_dict(model_dict)  # load the new state dict
params = "{}-{}-batch{}-epoch{}-lr{}-momentum{}-wdecay{}".format(args.target_dataset, args.optimizer, args.batch_size, args.epochs, args.lr, args.momentum, args.w_decay)

# use GPU
if args.cuda:
  ts = time.time()
  model = model.cuda()
  if args.multi_gpu:
    num_gpu = list(range(torch.cuda.device_count()))
    model = nn.DataParallel(model, device_ids=num_gpu)
  print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

# define loss & optimizer
if args.optimizer == 'Adam':
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
elif args.optimizer == 'SGD':
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
elif args.optimizer == 'RMSprop':
  optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.01)
criterion = nn.BCEWithLogitsLoss()


def train(epoch):
  model.train()
  for idx, batch in enumerate(train_loader):
    optimizer.zero_grad()
    if args.cuda:
      batch['X'] = batch['X'].cuda()
      batch['Y'] = batch['Y'].cuda()
    inputs, target = Variable(batch['X']), Variable(batch['Y'])
    output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if idx % args.log_interval == 0:
      print("Training epoch {}, idx {}, loss {}".format(epoch, idx, loss.data[0]))


def val(epoch):
  model.eval()
  val_loss = 0.
  correct = 0
  for idx, batch in enumerate(val_loader):
    if args.cuda:
      batch['X'] = batch['X'].cuda()
      batch['Y'] = batch['Y'].cuda()
    inputs, target = Variable(batch['X']), Variable(batch['Y'])
    output = model(inputs)
    val_loss += criterion(output, target).data[0]
    pred = np.argmax(output.data.cpu().numpy(), axis=1)
    target = np.argmax(target.data.cpu().numpy(), axis=1)
    correct += (pred == target).sum()

  val_loss /= idx
  acc = correct / len(val_data)
  accs[epoch] = acc
  np.save(os.path.join(result_dir, params), accs)
  if acc >= np.max(accs):
    model_name = os.path.join(model_dir, params)
    torch.save(model, model_name)
  print("Validating epoch {}, val_loss {}, acc {:.4f}({}/{})".format(epoch, val_loss, acc, correct, len(val_data)))


if __name__ == "__main__":
  val(0)  # test initial performance before training

  # training with one dataset
  print("Strat training")
  for epoch in range(args.epochs):
    scheduler.step()
    ts = time.time()
    train(epoch)
    val(epoch)
    print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
  print("Best val acc {}".format(np.max(accs)))
