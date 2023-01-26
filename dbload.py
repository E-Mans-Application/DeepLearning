"""
load corresponding databases
Adapted from
https://github.com/dataflowr/notebooks/blob/master/Module6/06_convolution_digit_recognizer.ipynb
"""


#import math,sys,os,numpy as np
from typing import Tuple
#from numpy.linalg import norm
#from matplotlib import pyplot as plt

#import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import ToTensor
#from torchvision import models,transforms,datasets

def load_dataset(set_provider, root_dir, batch_size=1, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """Loads a dataset. 
    * `set_cls` is a callable that takes two keyword arguments: a string `root` and a
    bool `train` (indicating whether to load the training or the test set), and returning a Dataset located in the
    given root dir. Required.
    * `root_dir` is the root folder in which the dataset will be downloaded. Required.
    * `**kwargs` see `torchvision.utils.data.DataLoader` for the list of additional arguments
        
    # Returns:
    (train_loader : `torchvision.utils.data.DataLoader`,
    test_loader : `torchvision.utils.data.DataLoader`)"""

    train_set = set_provider(root=root_dir, train=True)
    test_set = set_provider(root=root_dir, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, **kwargs)
    test_loader = DataLoader(test_set, batch_size=len(test_set), **kwargs)

    return train_loader, test_loader

def load_mnist(download=True, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """Loads MNIST dataset. See `torchvision.utils.data.DataLoader` for the list of accepted
    arguments.
    
    # Returns:
    (train_loader, test_loader)"""

    def provider(root, train):
        return torchvision.datasets.MNIST(root=root, train=train, download=True, transform=ToTensor())
    return load_dataset(provider, './data/MNIST/', shuffle=True, num_workers=1, **kwargs)