'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.normed = False

    def norm(self):
        self.params_for_norm = []
        modules = list(self.children())
        for idx, l in enumerate(modules, 0):
            if isinstance(l, nn.Conv2d):
                self.params_for_norm.append(conv(l).reshape(-1, 1, 1, 1))

            elif isinstance(l, nn.BatchNorm2d):
                self.params_for_norm.append(bn(l).reshape(-1, 1, 1, 1))

            elif isinstance(l, nn.Sequential):
                if len(list(l.children())) > 0:
                    self.params_for_norm.append(conv(l[0]).reshape(-1, 1, 1, 1))
                    self.params_for_norm.append(bn(l[1]).reshape(-1, 1, 1, 1))

        if torch.cuda.is_available():
            for p in self.params_for_norm:
                p.cuda()

        self.normed = True

    def forward(self, x):
        out = self.conv1(x)
        if self.normed is True:
            idx = 0
            out = F.conv2d(out, self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
            idx += 1
        out = self.bn1(out)
        if self.normed is True:
            out = F.conv2d(out, self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
            idx += 1
        out = F.relu(out)

        out = self.conv2(out)
        if self.normed is True:
            out = F.conv2d(out, self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
            idx += 1
        out = self.bn2(out)
        if self.normed is True:
            out = F.conv2d(out, self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
            idx += 1

        if len(list(self.shortcut.children())) > 0 and self.normed is True:
            x = F.conv2d(self.shortcut[0](x), self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
            idx += 1
            out += F.conv2d(self.shortcut[1](x), self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
            idx += 1
        else:
            out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, width, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = width

        self.conv1 = nn.Conv2d(3, width, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*width, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*width*block.expansion, num_classes)
        self.normed = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.normed is True:
            idx = 0
            out = F.conv2d(out, self.params_for_norm[idx], bias=None, groups= self.params_for_norm[idx].shape[0])
            idx += 1
        out = self.bn1(out)
        if self.normed is True:
            out = F.conv2d(out, self.params_for_norm[idx], bias=None, groups= self.params_for_norm[idx].shape[0])
            idx += 1
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.normed is True:
            out = F.linear(out, self.params_for_norm[idx], bias=None)
            idx += 1

        return out

    def norm(self):
        self.params_for_norm = []
        modules = list(self.children())
        for idx, l in enumerate(modules, 0):
            if isinstance(l, nn.Conv2d):
                self.params_for_norm.append(conv(l).reshape(-1, 1, 1, 1))

            elif isinstance(l, nn.BatchNorm2d):
                self.params_for_norm.append(bn(l).reshape(-1, 1, 1, 1))

            elif isinstance(l, nn.Sequential):
                for l in l.children():
                    l.norm()

            elif isinstance(l, nn.Linear):
                p = l.weight.data.norm(dim=1, keepdim=True) + l.bias.data.abs().view(-1, 1)
                s = l.weight.data.shape
                l.weight.data.view(s[0], -1).div_(p).view(s)
                l.bias.data.div_(p.reshape(-1))
                self.params_for_norm.append(torch.diag(p.reshape(-1)))

        self.normed = True
        if torch.cuda.is_available():
            for p in self.params_for_norm:
                p.cuda()


def conv(l):
    s = l.weight.data.shape
    p = l.weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
    l.weight.data.view(s[0], -1).div_(p).view(s)
    return p


def bn(l):
    p = l.weight.data.abs().view(-1, 1) + l.bias.data.abs().view(-1,1)
    s = l.weight.data.shape
    l.weight.data.view(s[0], -1).div_(p).view(s)
    l.bias.data.div_(p.reshape(-1))
    return p


def resnet_dd(width):
    return ResNet(BasicBlock, [2, 2, 2, 2], width)


if __name__ == "__main__":

    model = resnet_dd(2)
    x = torch.randn(1, 3, 32, 32)
    print(model(x))
    # model.norm()
    # print(model(x))
