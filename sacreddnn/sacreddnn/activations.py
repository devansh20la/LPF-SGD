import torch
import torch.nn as nn
import torch.nn.functional as F

def requ(input):
    x = F.relu(input) 
    return x * x

class ReQU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return requ(input)

def quadu(input):
    return input*input

class QuadU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return quadu(x)

def swish(x):
    return x * torch.sigmoid(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return swish(x)

def mish(x):
    return x * torch.tanh(F.softplus(x))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)

def quadlin(x):
    return torch.min(torch.abs(x), x*x)

class QuadLin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return quadlin(x)
    
def quadbald(x, k=2):
    return nn.functional.threshold(2*k*torch.abs(x)-k*k, k*k, 0, inplace=True) - nn.functional.threshold(-x*x, -k*k, 0, inplace=True)

class QuadBald(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, k=2):
        #return nn.functional.threshold(2*k*torch.abs(x)-k*k, k*k, 0, inplace=True) - nn.functional.threshold(-x*x, -k*k, 0, inplace=True)
        return quadbald(x, k=k)

def slowquad(x, k=2):
    return nn.functional.threshold(torch.abs(x)-1, k-1, 0, inplace=True) - nn.functional.threshold(-x*x/4., -k*k/4, 0, inplace=True)

class SlowQuad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, k=2):
        return slowquad(x, k=k)

    
# Replace Relu activations with new_act (e.g new_act = Swish())
# See https://discuss.pytorch.org/t/change-all-conv2d-and-batchnorm2d-to-their-3d-counterpart/24780 for changing network layers
def replace_relu_(model, new_act):
    if isinstance(new_act, str):
        new_act =   nn.ReLU() if new_act == 'relu' else\
                    nn.Tanh() if new_act == 'tanh' else\
                    ReQU() if new_act == 'requ' else\
                    QuadU() if new_act == 'quadu' else\
                    Swish() if new_act == 'swish' else\
                    QuadLin() if new_act == 'quadlin' else\
                    QuadBald() if new_act == 'quadbald' else\
                    SlowQuad() if new_act == 'slowquad' else\
                    Mish() if new_act == 'mish' else None
    if new_act == None:
        return

    new_modules = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            print("Replacing ReLU")
            new_modules[name] = new_act

    for name in new_modules:
        parent_module = model
        objs = name.split(".")
        for obj in objs[:-1]:
            parent_module = parent_module.__getattr__(obj)
        parent_module.__setattr__(objs[-1], new_modules[name])