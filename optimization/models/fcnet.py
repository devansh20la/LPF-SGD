import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def norm(self):
        self.params_for_norm = []
        s = self.conv1.weight.data.shape
        p = self.conv1.weight.data.view(s[0], -1).norm(dim=1, keepdim=True) + self.conv1.bias.data.abs().view(-1, 1)
        self.conv1.weight.data.view(s[0], -1).div_(p)
        self.conv1.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(p.reshape(-1, 1, 1, 1))

        s = self.conv2.weight.data.shape
        p = self.conv2.weight.data.view(s[0], -1).norm(dim=1, keepdim=True) + self.conv2.bias.data.abs().view(-1, 1)
        self.conv2.weight.data.view(s[0], -1).div_(p)
        self.conv2.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(p.reshape(-1, 1, 1, 1))

        s = self.fc1.weight.data.shape
        p = self.fc1.weight.data.norm(dim=1, keepdim=True) + self.fc1.bias.data.abs().view(-1, 1)
        self.fc1.weight.data.view(s[0], -1).div_(p)
        self.fc1.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(torch.diag(p.reshape(-1)))

        s = self.fc2.weight.data.shape
        p = self.fc2.weight.data.norm(dim=1, keepdim=True) + self.fc2.bias.data.abs().view(-1, 1)
        self.fc2.weight.data.view(s[0], -1).div_(p)
        self.fc2.bias.data.div_(p.reshape(-1))
        self.params_for_norm.append(torch.diag(p.reshape(-1)))

        def new_forward(x):
            x = F.conv2d(self.conv1(x), self.params_for_norm[0], groups=self.params_for_norm[0].shape[0], bias=None)
            x = F.relu(x)
            x = F.conv2d(self.conv2(x), self.params_for_norm[1], groups=self.params_for_norm[1].shape[0], bias=None)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = F.linear(self.fc1(x), self.params_for_norm[2], bias=None)
            x = F.relu(x)
            x = self.dropout2(x)
            x = F.linear(self.fc2(x), self.params_for_norm[3], bias=None)
            return x

        self.forward = new_forward

if __name__ == "__main__":
    x = torch.randn(1,1,28,28)
    model = LeNet(10)
    model.eval()
    y = model(x)

    print(y)
    model.norm()
    y2 = model(x)
    print(y2)


