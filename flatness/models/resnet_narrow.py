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

    def __init__(self, args, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.batchnorm = args.batchnorm
        self.skip = args.skip

        if self.batchnorm:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        if self.skip:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                if self.batchnorm:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                        nn.BatchNorm2d(self.expansion*planes)
                    )
                else:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
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
                if len(list(l.children())) == 2:
                    self.params_for_norm.append(conv(l[0]).reshape(-1, 1, 1, 1))
                    self.params_for_norm.append(bn(l[1]).reshape(-1, 1, 1, 1))
                elif len(list(l.children())) == 1:
                    self.params_for_norm.append(conv(l[0]).reshape(-1, 1, 1, 1))

        if torch.cuda.is_available():
            for p in self.params_for_norm:
                p.cuda()

        self.normed = True

    def forward(self, x):
        out = self.conv1(x)
        if self.normed is True:
            idx=0
            out = F.conv2d(out, self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
            idx+=1
        if self.batchnorm:
            out = self.bn1(out)
            if self.normed is True:
                out = F.conv2d(out, self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
                idx+=1
        out = F.relu(out)

        out = self.conv2(out)
        if self.normed is True:
            out = F.conv2d(out, self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
            idx+=1
        if self.batchnorm:
            out = self.bn2(out)
            if self.normed is True:
                out = F.conv2d(out, self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
                idx+=1

        if self.skip:
            if len(list(self.shortcut.children())) > 0 and self.normed is True:
                if self.batchnorm:
                    x = F.conv2d(self.shortcut[0](x), self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
                    idx += 1
                    out += F.conv2d(self.shortcut[1](x), self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
                    idx += 1
                else:
                    out += F.conv2d(self.shortcut(x), self.params_for_norm[idx], groups=self.params_for_norm[idx].shape[0], bias=None)
                    idx += 1
            else:
                out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.args = args
        self.in_planes = self.args.width * 8

        if self.args.batchnorm:
            self.conv1 = nn.Conv2d(3, self.args.width * 8, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(self.args.width * 8)
        else:
            self.conv1 = nn.Conv2d(3, self.args.width * 8, kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, self.args.width * 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.args.width * 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.args.width * 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.args.width * 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(self.args.width * 64 * block.expansion, num_classes)
        self.normed = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.args, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.normed is True:
            idx = 0
            out = F.conv2d(out, self.params_for_norm[idx], bias=None, groups= self.params_for_norm[idx].shape[0])
            idx += 1
        if self.args.batchnorm:
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


def resnet18_narrow(args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], args, **kwargs)


def conv(l):
    s = l.weight.data.shape
    p = l.weight.data.view(s[0], -1).norm(dim=1, keepdim=True) + l.bias.data.abs().view(-1, 1)
    l.weight.data.view(s[0], -1).div_(p).view(s)
    l.bias.data.div_(p.reshape(-1))
    return p


def bn(l):
    p = l.weight.data.abs().view(-1, 1) + l.bias.data.abs().view(-1,1)
    s = l.weight.data.shape
    l.weight.data.view(s[0], -1).div_(p).view(s)
    l.bias.data.div_(p.reshape(-1))
    return p


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from args import get_args

    args = get_args(["--exp_num", "900", "--dtype", "cifar10"])
    model = resnet18_narrow(args)
    x = torch.randn(1, 3, 32, 32)
    print(model(x))
    model.norm()
    print(model(x))


