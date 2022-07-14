import torch
from torchvision.models import resnet18 

model = resnet18()

def new_func(a):
	return a.data.add_(torch.empty_like(a, device=a.data.device).normal_(0,1))
	
new_list = list(map(lambda a: , model.parameters()))
print(model)