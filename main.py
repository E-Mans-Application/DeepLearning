from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import dbload

class ForwardPreHook:
    """Hook to apply pruning. See PyTorch doc about `Module`'s "forward pre-hooks" """

    def __call__(self, module: nn.Module, *args) -> None:
        for param in module.parameters():
            param.data[param.mask == 0] = 0

class Prunable(nn.Module):
    """Base class for all prunable `Module`s.
    This class works the same way as the base `Module`,
    with the difference it has a method `prune_` added."""

    def __init__(self):
        super().__init__()
        self._hook = self.register_forward_pre_hook(ForwardPreHook())

    def _create_mask(tensor: torch.Tensor) -> None:
        """Initializes a mask, filled with ones, for the given tensor."""
        tensor.mask = torch.ones_like(tensor)

    def __setattr__(self, name: str, value: Union[torch.Tensor, nn.Module]) -> None:
        # Initializes a mask for parameters and sub-modules
        if isinstance(value, nn.Module):
            for param in value.parameters():
                Prunable._create_mask(param)
        elif isinstance(value, torch.Tensor):
            Prunable._create_mask(value)
        return super().__setattr__(name, value)
        
    def prune_(self, perc : float) -> None:
        """Prunes the model (in-place), disabling `perc * 100` percents of
        the weights.
        For each parameter, the weights with the lowest absolute value are pruned.
        `float` is a number between 0 and 1.
        
        ### Note:
        Pruning does not enforce the gradient to be null in the backward step.
        It is only applied during evaluation."""
        for param in self.parameters():
            q = torch.quantile(abs(param), perc).item()
            param.mask[abs(param) < q] = 0

        # forces to apply hook (allows to immediately reflect the changes
        # in tests)
        ForwardPreHook()(self)

class LeNet(Prunable):
    """Prunable LeNet model
    
    Inspired from
    https://github.com/lychengrex/LeNet-5-Implementation-Using-Pytorch/blob/master/LeNet-5%20Implementation%20Using%20Pytorch.ipynb"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    print(next(iter(dbload.load_mnist()[0])))

    def test_prune(m: Prunable):
            
        z_g = 0
        c_g = 0

        for name, param in m.named_parameters():
            z = torch.sum((param == 0).int())
            c = torch.numel(param)

            z_g += z
            c_g += c

            print(name, ": ", z / c)

        print("Global: ", z_g / c_g)

    m = LeNet()
    print("Proportion of nulls (pruned) before:")
    test_prune(m)

    m.prune_(0.7)
    print("Proportion of nulls (pruned) after:")
    test_prune(m)
