import torch
import torch.nn as nn
from models import cifar_resnet18
import copy


model = cifar_resnet18()
noise = []
for mp in model.parameters():
    temp = torch.empty_like(mp, device=mp.device)
    temp.normal_(0, 1)
    temp.requires_grad = True
    noise.append(temp)


@torch.no_grad()
def check(noise):
    noise = list(map(lambda x: x*10, noise))


print(noise[0][0,0])
check(noise)
print(noise[0][0,0])
quit()
criterion = nn.CrossEntropyLoss()
y = torch.zeros(2).type(torch.LongTensor)
x = torch.randn(2, 3, 32, 32)

with torch.no_grad():
    for mp, n in zip(model.parameters(), noise):
        mp.data.add_(n)
batch_loss = criterion(model(x), y)
batch_loss.backward()

with torch.no_grad():
    for mp, n in zip(model.parameters(), noise):
        mp.data.sub_(n)
        n.grad = copy.deepcopy(mp.grad)
print(noise[0].grad)
