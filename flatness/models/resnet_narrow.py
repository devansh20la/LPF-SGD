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
        self.regular = args.regular
        self.skip = args.skip

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if self.regular == 'batch_norm':
            self.bn1 = nn.BatchNorm2d(planes)

        elif self.regular == "dropout":
            self.bn1 = nn.Dropout(0.2, inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if self.regular == 'batch_norm':
            self.bn2 = nn.BatchNorm2d(planes)
        elif self.regular == "dropout":
            self.bn2 = nn.Dropout(0.2, inplace=True)

        if self.skip:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                if self.regular == "batch_norm":
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion*planes)
                    )
                else:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                        )

    def forward(self, x):
        out = self.conv1(x)
        if self.regular == "batch_norm":
            out = self.bn1(out)
        out = F.relu(out)
        if self.regular == "dropout":
            out = self.bn1(out)

        out = self.conv2(out)
        if self.regular == "batch_norm":
            out = self.bn2(out)

        if self.skip:
            out += self.shortcut(x)
        out = F.relu(out)
        if self.regular == "dropout":
            out = self.bn2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.args = args
        self.in_planes = self.args.width * 8

        self.conv1 = nn.Conv2d(3, self.args.width * 8, kernel_size=3, stride=1, padding=1, bias=False)
        if self.args.regular == "batch_norm":
            self.bn1 = nn.BatchNorm2d(self.args.width * 8)
        elif self.args.regular == "dropout":
            self.bn1 = nn.Dropout(0.2, inplace=True)

        self.layer1 = self._make_layer(block, self.args.width * 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.args.width * 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.args.width * 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.args.width * 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(self.args.width * 64 * block.expansion, num_classes)

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
        if self.args.regular == "batch_norm":
            out = self.bn1(out)

        out = F.relu(out)

        if self.args.regular == "dropout":
            out = self.bn1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18_narrow(args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], args, **kwargs)


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from args import get_args

    args = get_args(["--exp_num", "250"])
    model = resnet18_narrow(args)

    print(args, model)
