# Shannon Entropy
import torch


def shannon_entropy(model_func):

	res = 0.0
	for inputs, targets in model_func.dataloader['train']:
		if model_func.use_cuda:
			inputs = inputs.cuda()
			targets = targets.cuda()

		with torch.no_grad():
			outputs = model_func.model(inputs)
			res += torch.sum(torch.nn.Softmax(1)(outputs) * torch.nn.LogSoftmax(1)(outputs))

	return (-res / len(model_func.dataloader['train'].dataset)).item()