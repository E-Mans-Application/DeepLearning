"""
load corresponding databases
Adapted from
https://github.com/dataflowr/notebooks/blob/master/Module6/06_convolution_digit_recognizer.ipynb
"""


import math,sys,os,numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision import models,transforms,datasets

root_dir = './data/MNIST/'
torchvision.datasets.MNIST(root=root_dir,download=True)


train_set = torchvision.datasets.MNIST(root=root_dir, train=True, download=True)
MNIST_dataset = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
