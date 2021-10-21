import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.normed = False

    def forward(self, x):
        out = F.relu(self.conv1(x))
        if self.normed is True:
            out = F.conv2d(out, self.params_for_norm[0], bias=None, groups=self.params_for_norm[0].shape[0])
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2(out))
        if self.normed is True:
            out = F.conv2d(out, self.params_for_norm[1], bias=None, groups=self.params_for_norm[1].shape[0])
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.normed is True:
            out = F.linear(out, self.params_for_norm[2], bias=None)
        out = F.relu(self.fc2(out))
        if self.normed is True:
            out = F.linear(out, self.params_for_norm[3], bias=None)
        out = self.fc3(out)
        if self.normed is True:
            out = F.linear(out, self.params_for_norm[4], bias=None)
        return out

    def norm(self):
        self.params_for_norm = []
        s = self.conv1.weight.data.shape
        p = self.conv1.weight.data.view(s[0], -1).norm(dim=1, keepdim=True) + self.conv1.bias.data.abs().view(-1, 1)
        self.conv1.weight.data.view(s[0], -1).div_(p).view(s)
        self.conv1.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(p.reshape(-1, 1, 1, 1))

        s = self.conv2.weight.data.shape
        p = self.conv2.weight.data.view(s[0], -1).norm(dim=1, keepdim=True) + self.conv2.bias.data.abs().view(-1, 1)
        self.conv2.weight.data.view(s[0], -1).div_(p).view(s)
        self.conv2.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(p.reshape(-1, 1, 1, 1))

        s = self.fc1.weight.data.shape
        p = self.fc1.weight.data.norm(dim=1, keepdim=True) + self.fc1.bias.data.abs().view(-1, 1)
        self.fc1.weight.data.view(s[0], -1).div_(p).view(s)
        self.fc1.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(torch.diag(p.reshape(-1)))

        s = self.fc2.weight.data.shape
        p = self.fc2.weight.data.norm(dim=1, keepdim=True) + self.fc2.bias.data.abs().view(-1, 1)
        self.fc2.weight.data.view(s[0], -1).div_(p).view(s)
        self.fc2.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(torch.diag(p.reshape(-1)))

        s = self.fc3.weight.data.shape
        p = self.fc3.weight.data.norm(dim=1, keepdim=True) + self.fc3.bias.data.abs().view(-1, 1)
        self.fc3.weight.data.view(s[0], -1).div_(p).view(s)
        self.fc3.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(torch.diag(p.reshape(-1)))

        self.normed = True


class OG_LeNet(nn.Module):
    def __init__(self):
        super(OG_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


if __name__ == "__main__":
    model = test_model()
    x = torch.randn(1,1,28,28)
    model(x)
    d = 0
    for p in model.parameters():
        d+=p.numel()
    print(d)



