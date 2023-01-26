if __name__ == "__main__":

	from main import *

	truc = Prunable()
	x = torch.tensor([[1., -1.], [1., -1.]])
	print("test1: ", truc.num_flat_features(x))

	print(next(iter(dbload.load_mnist()[0])))

	m = LeNet()
	print("Proportion of nulls (pruned) before:")
	test_prune(m)

	m.prune_(0.7)
	print("Proportion of nulls (pruned) after:")
	test_prune(m)
