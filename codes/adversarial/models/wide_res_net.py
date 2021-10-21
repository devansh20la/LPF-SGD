import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropRate=0.0):
        super(Wide_ResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BasicUnit(nn.Module):
#     def __init__(self, channels: int, dropout: float):
#         super(BasicUnit, self).__init__()
#         self.block = nn.Sequential(OrderedDict([
#             ("0_normalization", nn.BatchNorm2d(channels)),
#             ("1_activation", nn.ReLU(inplace=True)),
#             ("2_convolution", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
#             ("3_normalization", nn.BatchNorm2d(channels)),
#             ("4_activation", nn.ReLU(inplace=True)),
#             ("5_dropout", nn.Dropout(dropout, inplace=True)),
#             ("6_convolution", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
#         ]))

#     def norm(self):
#         self.block_params = []
#         for l in self.block.children():
#             if isinstance(l, nn.Conv2d):
#                 s = l.weight.data.shape
#                 p = l.weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
#                 l.weight.data.view(s[0], -1).div_(p).view(s)
#                 self.block_params.append(p.reshape(-1, 1, 1, 1))
#             elif isinstance(l, nn.BatchNorm2d):
#                 p = l.weight.data.abs().view(-1,1) + l.bias.data.abs().view(-1,1)
#                 s = l.weight.data.shape
#                 l.weight.data.view(s[0], -1).div_(p).view(s)
#                 l.bias.data.div_(p.reshape(-1))
#                 self.block_params.append(p.reshape(-1, 1, 1, 1))

#         def new_forward(x):
#             r = 0
#             for l in self.block.children():
#                 x = l(x)
#                 if isinstance(l, nn.Conv2d) or isinstance(l, nn.BatchNorm2d):
#                     x = F.conv2d(x, self.block_params[r], groups=self.block_params[r].shape[0], bias=None)
#                     r+=1
#             return x

#         self.forward = new_forward

#     def forward(self, x):
#         return self.block(x)


# class DownsampleUnit(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
#         super(DownsampleUnit, self).__init__()
#         self.norm_act = nn.Sequential(OrderedDict([
#             ("0_normalization", nn.BatchNorm2d(in_channels)),
#             ("1_activation", nn.ReLU(inplace=True)),
#         ]))
#         self.block = nn.Sequential(OrderedDict([
#             ("0_convolution", nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)),
#             ("1_normalization", nn.BatchNorm2d(out_channels)),
#             ("2_activation", nn.ReLU(inplace=True)),
#             ("3_dropout", nn.Dropout(dropout, inplace=True)),
#             ("4_convolution", nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)),
#         ]))
#         self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False)
    
#     def norm(self):
#         self.norm_act_params = []
#         for l in self.norm_act.children():
#             if isinstance(l, nn.BatchNorm2d):
#                 p = l.weight.data.abs().view(-1,1) + l.bias.data.abs().view(-1,1)
#                 s = l.weight.data.shape
#                 l.weight.data.view(s[0], -1).div_(p).view(s)
#                 l.bias.data.div_(p.reshape(-1))
#                 self.norm_act_params.append(p.reshape(-1, 1, 1, 1))
        
#         self.block_params = []
#         for l in self.block.children():
#             if isinstance(l, nn.Conv2d):
#                 s = l.weight.data.shape
#                 p = l.weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
#                 l.weight.data.view(s[0], -1).div_(p).view(s)
#                 self.block_params.append(p.reshape(-1, 1, 1, 1))
#             elif isinstance(l, nn.BatchNorm2d):
#                 p = l.weight.data.abs().view(-1,1) + l.bias.data.abs().view(-1,1)
#                 s = l.weight.data.shape
#                 l.weight.data.view(s[0], -1).div_(p).view(s)
#                 l.bias.data.div_(p.reshape(-1))
#                 self.block_params.append(p.reshape(-1, 1, 1, 1))

#         self.downsample_params = []
#         s = self.downsample.weight.data.shape
#         p = self.downsample.weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
#         self.downsample.weight.data.view(s[0], -1).div_(p)
#         self.downsample_params.append(p.reshape(-1, 1, 1, 1))

#         def new_forward(x):
#             for l in self.norm_act.children():
#                 x = l(x)
#                 if isinstance(l, nn.BatchNorm2d):
#                     x = F.conv2d(x, self.norm_act_params[0], groups=self.norm_act_params[0].shape[0], bias=None)
#             x1 = None
#             r = 0
#             for l in self.block.children():
#                 if x1 is None:
#                     x1 = l(x)
#                 else:
#                     x1 = l(x1)
#                 if isinstance(l, nn.Conv2d) or isinstance(l, nn.BatchNorm2d):
#                     x1 = F.conv2d(x1, self.block_params[r], groups=self.block_params[r].shape[0], bias=None)
#                     r+=1
#             r = 0
#             x = self.downsample(x)
#             x = F.conv2d(x, self.downsample_params[r], groups=self.downsample_params[r].shape[0], bias=None)
#             return x1 + x

#         self.forward = new_forward

#     def forward(self, x):
#         x = self.norm_act(x)
#         return self.block(x) + self.downsample(x)


# class Block(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float):
#         super(Block, self).__init__()
#         self.block = nn.Sequential(
#             DownsampleUnit(in_channels, out_channels, stride, dropout),
#             *(BasicUnit(out_channels, dropout) for _ in range(depth))
#         )

#     def norm(self):
#         for l in self.block.children():
#             l.norm()

#     def forward(self, x):
#         return self.block(x)


# class Wide_ResNet(nn.Module):
#     def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, labels: int):
#         super(Wide_ResNet, self).__init__()

#         self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
#         self.block_depth = (depth - 4) // (3 * 2)

#         self.f = nn.Sequential(OrderedDict([
#             ("0_convolution", nn.Conv2d(in_channels, self.filters[0], (3, 3), stride=1, padding=1, bias=False)),
#             ("1_block", Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout)),
#             ("2_block", Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout)),
#             ("3_block", Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout)),
#             ("4_normalization", nn.BatchNorm2d(self.filters[3])),
#             ("5_activation", nn.ReLU(inplace=True)),
#             ("6_pooling", nn.AvgPool2d(kernel_size=8)),
#             ("7_flattening", nn.Flatten()),
#             ("8_classification", nn.Linear(in_features=self.filters[3], out_features=labels)),
#         ]))

#         self._initialize()

#     def _initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.zero_()
#                 m.bias.data.zero_()

#     def norm(self):
#         self.params_for_norm = []

#         for l in self.f.children():
#             if isinstance(l, nn.Conv2d):
#                 s = l.weight.data.shape
#                 p = l.weight.data.view(s[0], -1).norm(dim=1, keepdim=True)
#                 l.weight.data.view(s[0], -1).div_(p)
#                 self.params_for_norm.append(p.reshape(-1, 1, 1, 1))
#             if isinstance(l, nn.BatchNorm2d):
#                 p = l.weight.data.abs().view(-1,1) + l.bias.data.abs().view(-1,1)
#                 s = l.weight.data.shape
#                 l.weight.data.view(s[0], -1).div_(p)
#                 l.bias.data.div_(p.reshape(-1))
#                 self.params_for_norm.append(p.reshape(-1, 1, 1, 1))
#             elif isinstance(l, Block):
#                 l.norm()

#         def new_forward(x):
#             for l in self.f.children():
#                 x = l(x)
#                 if isinstance(l, nn.Conv2d):
#                     x = F.conv2d(x, self.params_for_norm[0], groups=self.params_for_norm[0].shape[0], bias=None)
#                 if isinstance(l, nn.BatchNorm2d):
#                     x = F.conv2d(x, self.params_for_norm[1], groups=self.params_for_norm[1].shape[0], bias=None)
#             return x

#         self.forward = new_forward

#     def forward(self, x):
#         return self.f(x)

if __name__ == "__main__":
    model = Wide_ResNet(28, 10, num_classes=10)
    print(model)

