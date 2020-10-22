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

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if self.batchnorm:
            self.bn2 = nn.BatchNorm2d(planes)

        if self.skip:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                if self.batchnorm:
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
        if self.batchnorm:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        if self.skip:
            out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.args = args
        self.in_planes = self.args.width * 8

        self.conv1 = nn.Conv2d(3, self.args.width * 8, kernel_size=3, stride=1, padding=1, bias=False)
        if self.args.batchnorm:
            self.bn1 = nn.BatchNorm2d(self.args.width * 8, track_running_stats=False)

        # self.layer1 = self._make_layer(block, self.args.width * 8, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, self.args.width * 16, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, self.args.width * 32, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, self.args.width * 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(25088, num_classes)

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

        if self.args.batchnorm:
            out = self.bn1(out)

        out = F.relu(out)

        # out = self.layer1(out)
        # # out = self.layer2(out)
        # # out = self.layer3(out)
        # # out = self.layer4(out)
        # # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18_narrow(args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], args, **kwargs)


def resnet18_test(args, **kwargs):
    return ResNet(BasicBlock, [1], args, **kwargs)


def modify_any_way_you_like(param):
    param.div_(param.norm())


def modify_and_get_right_gamma(weight, bias):
    gamma = weight.norm() + bias.norm()
    weight.div_(gamma)
    bias.div_(gamma)
    return gamma


def modify_with_right_norm(weight, gamma):
    weight.mul_(gamma)
    gamma = weight.norm()
    weight.div_(gamma)
    return gamma


def handle_seq_with_identity_skip(layer):
    mods = list(layer.named_modules())[2:]

    modify_any_way_you_like(mods[0][1].weight.data)
    right_norm = modify_and_get_right_gamma(mods[1][1].weight.data, mods[1][1].bias.data)
    modify_with_right_norm(mods[2][1].weight.data, right_norm)

    right_norm = modify_and_get_right_gamma(mods[3][1].weight.data, mods[3][1].bias.data)

    return right_norm


if __name__ == "__main__":
    torch.random.manual_seed(1)
    for _ in range(100):
        import sys
        sys.path.append("..")
        from args import get_args
        # from utils import ENorm

        args = get_args(["--exp_num", "0"])
        model = resnet18_test(args)

        x = torch.randn(1, 3, 28, 28)
        y = model(x)
        # y = torch.nn.Softmax(dim=1)(y).argmax().item()

        layers = list(model.children())
        right_norm = None

        for i in range(len(layers)):
            layer = layers[i]
            if isinstance(layer, nn.Conv2d) and isinstance(layers[i+1], nn.BatchNorm2d):
                if right_norm is not None:
                    modify_with_right_norm(layer.weight, right_norm)
                    right_norm = None
                else:
                    modify_any_way_you_like(layer.weight.data)

            if isinstance(layer, nn.BatchNorm2d):
                right_norm = modify_and_get_right_gamma(layer.weight.data, layer.bias.data)

            # if isinstance(layer, nn.Sequential):
            #     right_norm = handle_seq_with_identity_skip(layer)

            if isinstance(layer, nn.Linear):
                layer.weight.data.mul_(right_norm)
                gamma = layer.weight.data.norm() + layer.bias.data.norm()
                layer.weight.data.div_(gamma)
                layer.bias.data.div_(gamma)

        print(torch.nn.Softmax(dim=1)(y).argmax().item() - torch.nn.Softmax(dim=1)(model(x)).argmax().item())
    quit()
