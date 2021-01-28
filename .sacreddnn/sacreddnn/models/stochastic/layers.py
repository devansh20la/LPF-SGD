import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import random
# import sys
# sys.path.append("..")
# from mlp import MLP


## https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html


class StochLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, noise_scale=1):
        super(StochLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.noise_scale = noise_scale
        self.mean = nn.Linear(in_features, out_features, bias)
        self.logstd = Parameter(torch.Tensor(out_features, in_features))
        self.logvar = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.mean, a=math.sqrt(5))
        
        # init each standard dev proportional to the mean
        with torch.no_grad():
            self.logstd.copy_(torch.log(self.noise_scale * torch.abs(self.mean.weight)))
            self.logvar.copy_(torch.log(self.noise_scale**2 * torch.abs(self.mean.weight**2)))
        
    def forward(self, x):
        z = self.mean.forward(x)
        # var = torch.exp(self.logstd)**2
        var = torch.exp(self.logvar)   # TODO : USE SOFTPLUS?
        sigma = torch.sqrt(F.linear(x**2, var))
#        # print(f"z={torch.norm(x)} eps={torch.norm(eps * sigma)}")
        eps = z.clone().detach_().normal_()
        if random.random() < 0.01:
            print(f"norm_var :{torch.norm(var)} norm_sigma:{torch.norm(sigma)}")
        return z + eps * sigma

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# class StochLinear(nn.Module):
#     def __init__(self, input_features, output_features, bias=True, noise_scale=0.1,
#             ):
#         super(StochLinear, self).__init__()
#         self.means = nn.Linear(input_features, output_features, bias=bias)
#         self.vars = nn.Linear(input_features, output_features, bias=False)
#         # self.act = nn.ReLU()

#         with torch.no_grad():
#             self.vars.weight.copy_(noise_scale**2 * self.vars.weight**2)
#             # torch.nn.init.constant_(tensor, val) # sqrt(2/(in +out))

#     def forward(self, x):
#         z = self.means.forward(x)
#         sigma =  torch.sqrt(self.vars.forward(x*x) + 10) # + 1e-6 inside sqrt for num stability?
#         eps = z.clone().detach_().normal_()
#         # print(f"z={torch.norm(x)} eps={torch.norm(eps * sigma)}")
#         return z + eps * sigma


# def make_stochastic(model):
#     if isinstance(model, MLP):
#         for (i, layer) in enumerate(model.layers):
#             if isinstance(layer, nn.Linear):
#                 model.layers[i] = StochLinear(layer.in_features, layer.out_features)
#     else:
#         raise Exception('cannot make this model stochastic')
#     return model


class StochMLP(nn.Module):
    def __init__(self, nin, nhs, nout):
        super(StochMLP, self).__init__()
        assert isinstance(nhs, list)

        self.layers = nn.Sequential(
            StochLinear(nin, nhs[0]),
            nn.ReLU(),
            *[StochLinear(nhs[i//2], nhs[i//2 + 1]) if i%2==0 else nn.ReLU() for i in range(2*(len(nhs)-1))],
            nn.Linear(nhs[-1], nout))
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.layers(x)

# test
if __name__ == '__main__':
    nin = 10
    nout = 1
    M = 2
    x = torch.randn(M, nin)
    y = torch.randn(M)
    model = StochMLP(nin, [10], nout)
    yhat = model(x)
    l = torch.nn.functional.mse_loss(yhat, y)
    l.backward()
    print("ciao")
