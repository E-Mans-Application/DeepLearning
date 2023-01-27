

import sys
import pathlib


from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import dbload
import visu


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
		super(Prunable, self).__init__()
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
	
	def num_flat_features(self, x):	# https://discuss.pytorch.org/t/understand-nn-module/8416
		
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
	
	def prune_(self, perc : float) -> None:
		"""Prunes the model (in-place), disabling `perc * 100` percents of
		the weights.
		For each parameter, the weights with the lowest absolute value are pruned.
		`float` is a number between 0 and 1.
		
		### Note:
		Pruning does not enforce the gradient to be null in the backward step.
		It is only applied during evaluation."""
		for param in self.parameters():
			q = torch.quantile(abs(param[param.mask == 1]), perc).item()
			param.mask[abs(param) < q] = 0

		# forces to apply hook (allows to immediately reflect the changes
		# in tests)
		ForwardPreHook()(self)

class LeNet(Prunable):
	"""Prunable LeNet model
	
	Inspired from
	https://github.com/lychengrex/LeNet-5-Implementation-Using-Pytorch/blob/master/LeNet-5%20Implementation%20Using%20Pytorch.ipynb"""

	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
		x = x.view(-1, super().num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def reset_parameters(self):
		"""Resets the parameters using random values, but retains
		the pruning status of the weights."""
		for module in self.children():
			module.reset_parameters()
		ForwardPreHook()(self)


def visualize_params(a_model, title = None):
	
	weights = {}
	biases = {}
	
	for name, param in a_model.named_parameters():
		
		if title is not None:
			if "weight" in name:
				weights[name.removesuffix(".weight")] = param.detach().numpy()
			if "bias" in name:
				biases[name.removesuffix(".bias")] = param.detach().numpy()
	
	if title is not None:
		visu.show_params(weights, biases, title, top_outputs_path)


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


def training_loop (epochs, net : Prunable, tuple_dataloader, prune_interval, prune_factor):
	#cf	 https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	
	with tqdm(range(epochs)) as epoch:

		while epoch.n < epochs:
			for data in tuple_dataloader[0]:
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data
			
				# zero the parameter gradients
				optimizer.zero_grad()
			
				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				
				if epoch.n % 2000 == 0:	# print every 2000 mini-batches
					validation(net, tuple_dataloader)

				epoch.update()
	
				if epoch.n >= epochs:
					break

				if prune_interval > 0 and epoch.n % prune_interval == 0:
					net.prune_(prune_factor)
					test_prune(net)


def validate(net, dataloader, desc):
	#cf	 https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	
	correct = 0
	total = 0
	for data in tqdm(dataloader, desc=desc):
		images, labels = data
		# calculate outputs by running images through the network
		outputs = net(images)
		# the class with the highest energy is what we choose as prediction
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	return 100 * correct // total


def validation(net, tuple_dataloader):
	
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		acc = validate(net, tuple_dataloader[1], desc="Validation (1/2)")		
		print(f'Accuracy of the network on the {len(tuple_dataloader[1]) * tuple_dataloader[1].batch_size} test images: {acc} %')

		acc = validate(net, tuple_dataloader[0], desc="Validation (2/2)")
		print(f'Accuracy of the network on the {len(tuple_dataloader[0]) * tuple_dataloader[0].batch_size} training images: {acc} %')

def calculate_accruracy_prunning(epochs, tuple_dataloader):
	"""
	return a dictionary where the key is the percentile of weights that have been prunned,
	and the value is the accuracy percent of the retrained (without prior randomisation)
	net
	"""
	res = {}
	for prunning_percent in range(0, 100, 5):
		model = LeNet()
		training_loop (epochs, model, mist_db, 500, float(prunning_percent)/ 100)
		res[prunning_percent] = validate(model,tuple_dataloader[1], desc="Validation calculate_accruracy_prunning")
	
	return res

if __name__ == "__main__":
	
	# CLI exploitation
	
	this_script_path = pathlib.Path(sys.argv[0])
	
	common_path = this_script_path.parent
	
	#this_script_name = this_script_path.stem
	
	# Storage management
	
	outputs_top_storage_name = "Outputs"
	top_outputs_path = common_path / outputs_top_storage_name
	top_outputs_path.mkdir(exist_ok = True)
	
	# Work
	
	mist_db = dbload.load_mnist(batch_size=10)
	model = LeNet()
	training_loop(10000, model, mist_db, 500, 0.1)
	visualize_params(model, "Trained full network prunned")
	validation(model, mist_db)

	print("Trying to train the pruned network from the beginning...")
	model.reset_parameters()
	test_prune(model)

	training_loop(10000, model, mist_db, -1, 0.1)
	test_prune(model)
	visualize_params(model, "Retrained prunned network")
	validation(model, mist_db)
	
	result = calculate_accruracy_prunning(5000, mist_db)
	
	print(result)
		
	visu.show_accuracy(result, top_outputs_path)
