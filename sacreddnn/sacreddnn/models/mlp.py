import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MLP(nn.Module):
    def __init__(self, nin, nhs, nout):
        super(MLP, self).__init__()
        assert isinstance(nhs, list)

        self.layers = nn.Sequential(
            nn.Linear(nin, nhs[0]),
            nn.ReLU(),
            *[nn.Linear(nhs[i//2], nhs[i//2 + 1]) if i%2==0 else nn.ReLU() for i in range(2*(len(nhs)-1))],
            nn.Linear(nhs[-1], nout))
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.layers(x)
