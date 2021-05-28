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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def norm(self):
        self.params_for_norm = []

        for l in self.children():
            if isinstance(l, nn.Conv2d):
                s = l.weight.data.shape
                p = l.weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
                l.weight.data.view(s[0], -1).div_(p)
                self.params_for_norm.append(p.reshape(-1, 1, 1, 1))
            elif isinstance(l, nn.BatchNorm2d):
                p = l.weight.data.abs().view(-1,1) + l.bias.data.abs().view(-1,1)
                s = l.weight.data.shape
                l.weight.data.view(s[0], -1).div_(p)
                l.bias.data.div_(p.reshape(-1))
                self.params_for_norm.append(p.reshape(-1, 1, 1, 1))
            elif isinstance(l, nn.Sequential):
                if len(list(l.children())) > 1:
                    s = l[0].weight.data.shape
                    p = l[0].weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
                    l[0].weight.data.view(s[0], -1).div_(p)
                    self.params_for_norm.append(p.reshape(-1, 1, 1, 1))

                    p = l[1].weight.data.abs().view(-1,1) + l[1].bias.data.abs().view(-1,1)
                    s = l[1].weight.data.shape
                    l[1].weight.data.view(s[0], -1).div_(p)
                    l[1].bias.data.div_(p.reshape(-1))
                    self.params_for_norm.append(p.reshape(-1, 1, 1, 1))

        def new_forward(x):
            out = F.conv2d(self.conv1(x), self.params_for_norm[0], bias=None, groups= self.params_for_norm[0].shape[0])
            out = F.conv2d(self.bn1(out), self.params_for_norm[1], bias=None, groups= self.params_for_norm[1].shape[0])
            out = F.relu(out)

            out = F.conv2d(self.conv2(out), self.params_for_norm[2], bias=None, groups= self.params_for_norm[2].shape[0])
            out = F.conv2d(self.bn2(out), self.params_for_norm[3], bias=None, groups= self.params_for_norm[3].shape[0])

            r = 4 
            for l in self.shortcut.children():
                x = F.conv2d(l(x), self.params_for_norm[r], bias=None, groups= self.params_for_norm[r].shape[0])
                r+=1

            out += x
            out = F.relu(out)
            return out

        self.forward = new_forward

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

    def norm(self):
        self.params_for_norm = []

        for l in self.children():
            if isinstance(l, nn.Conv2d):
                s = l.weight.data.shape
                p = l.weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
                l.weight.data.view(s[0], -1).div_(p)
                self.params_for_norm.append(p.reshape(-1, 1, 1, 1))
            elif isinstance(l, nn.BatchNorm2d):
                p = l.weight.data.abs().view(-1,1) + l.bias.data.abs().view(-1,1)
                s = l.weight.data.shape
                l.weight.data.view(s[0], -1).div_(p)
                l.bias.data.div_(p.reshape(-1))
                self.params_for_norm.append(p.reshape(-1, 1, 1, 1))
            elif isinstance(l, nn.Sequential):
                if len(list(l.children())) > 1:
                    s = l[0].weight.data.shape
                    p = l[0].weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
                    l[0].weight.data.view(s[0], -1).div_(p)
                    self.params_for_norm.append(p.reshape(-1, 1, 1, 1))

                    p = l[1].weight.data.abs().view(-1,1) + l[1].bias.data.abs().view(-1,1)
                    s = l[1].weight.data.shape
                    l[1].weight.data.view(s[0], -1).div_(p)
                    l[1].bias.data.div_(p.reshape(-1))
                    self.params_for_norm.append(p.reshape(-1, 1, 1, 1))

        def new_forward(x):
            out = F.conv2d(self.conv1(x), self.params_for_norm[0], bias=None, groups= self.params_for_norm[0].shape[0])
            out = F.conv2d(self.bn1(out), self.params_for_norm[1], bias=None, groups= self.params_for_norm[1].shape[0])
            out = F.relu(out)

            out = F.conv2d(self.conv2(out), self.params_for_norm[2], bias=None, groups= self.params_for_norm[2].shape[0])
            out = F.conv2d(self.bn2(out), self.params_for_norm[3], bias=None, groups= self.params_for_norm[3].shape[0])
            out = F.relu(out)

            out = F.conv2d(self.conv3(out), self.params_for_norm[4], bias=None, groups= self.params_for_norm[4].shape[0])
            out = F.conv2d(self.bn3(out), self.params_for_norm[5], bias=None, groups= self.params_for_norm[5].shape[0])

            r = 6 
            for l in self.shortcut.children():
                x = F.conv2d(l(x), self.params_for_norm[r], bias=None, groups= self.params_for_norm[r].shape[0])
                r+=1

            out += x
            out = F.relu(out)
            return out

        self.forward = new_forward


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def norm(self):
        self.params_for_norm = []
        for l in self.children():
            if isinstance(l, nn.Conv2d):
                s = l.weight.data.shape
                p = l.weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
                l.weight.data.view(s[0], -1).div_(p)
                self.params_for_norm.append(p.reshape(-1, 1, 1, 1))
            elif isinstance(l, nn.BatchNorm2d):
                p = l.weight.data.abs().view(-1,1) + l.bias.data.abs().view(-1,1)
                s = l.weight.data.shape
                l.weight.data.view(s[0], -1).div_(p).view(s)
                l.bias.data.div_(p.reshape(-1))
                self.params_for_norm.append(p.reshape(-1, 1, 1, 1))
            elif isinstance(l, nn.Sequential):
                for l in l.children():
                    l.norm()

        def new_forward(x):
            out = F.conv2d(self.conv1(x), self.params_for_norm[0], bias=None, groups= self.params_for_norm[0].shape[0])
            out = F.conv2d(self.bn1(out), self.params_for_norm[1], bias=None, groups= self.params_for_norm[1].shape[0])
            out = F.relu(out)

            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

        self.forward = new_forward

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


if __name__ == "__main__":
    net = ResNet50()
    net.eval()
    # net.load_state_dict(torch.load("../checkpoints/cifar10/resnet50/sgd/run_ms_0/best_model.pth.tar")['model'])

    x = torch.randn(1, 3, 32, 32)
    print(net(x))
    net.norm()
    print(net(x))

