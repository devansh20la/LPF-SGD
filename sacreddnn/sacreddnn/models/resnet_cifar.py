### This file contains the resnet models used in the paper
###  [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun,  Deep Residual Learning for Image Recognition. arXiv:1512.03385
### for Cifar10 experiments. 
### The models can also be used on Cifar100 and Imagener.

# """
# @title: ResNet in PyTorch for CIFAR-10
# References:
# ===========
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# [2] PyTorch Open Source Repository
#     https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# The plain/residual architectures follow the form in Fig. 3 (middle/right).
# The network inputs are 32×32 images, with the per-pixel mean subtracted.
# The first layer is 3×3 convolutions. Then we use a stack of 6n layers with
# 3×3 convolutions on the feature maps of sizes {32, 16, 8} respectively,
# with 2n layers for each feature map size.
# - The numbers of filters are {16, 32, 64} respectively.
# - The subsampling is performed by convolutions with a stride of 2.
# - The network ends with a global average pooling, a 10-way fully-connected layer, and softmax.
# - There are totally 6n+2 stacked weighted layers
#     (*) Expansion: # n_output_channels / n_input channels
# """

import torch
import torch.nn as nn
from torch.autograd import Variable

from ..activations import ReQU, QuadU, Swish, Mish, QuadLin, QuadBald, SlowQuad

def flatten(x): 
    return x.view(x.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class BasicBlock(nn.Module):

    expansion = 1       

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        #self.relu = QuadLin()
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):

        residue = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residue = self.downsample(x)

        out += residue
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = downsample
        
        self.stride = stride
        
    def forward(self, x):

        residue = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residue = self.downsample(x)

        out += residue
        out = self.relu(out)
        return out
            
            
class ResNet(nn.Module):

    def __init__(self, depth, name, num_classes=10, block=BasicBlock, activation=nn.ReLU()):
        super(ResNet, self).__init__()

        assert (depth - 2) % 6 == 0, 'Depth should be 6n + 2'
        n = (depth - 2) // 6

        self.name = name
        block = BasicBlock
        self.inplanes = 16
        fmaps = [16, 32, 64] # CIFAR10

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = QuadLin()

        self.layer1 = self._make_layer(block, fmaps[0], n, stride=1)
        self.layer2 = self._make_layer(block, fmaps[1], n, stride=2)
        self.layer3 = self._make_layer(block, fmaps[2], n, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten = flatten
        self.fc = nn.Linear(fmaps[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        ''' Between layers convolve input to match dimensions -> stride = 2 '''

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, print_sizes=False):
        
        if print_sizes:
            print('Sizes of the tensors inside each node: \n')
            print("\t In Model: input size", x.size())
        
        x = self.relu(self.bn(self.conv(x)))    # 32x32
        
        x = self.layer1(x)                      # 32x32
        x = self.layer2(x)                      # 16x16
        x = self.layer3(x)                      # 8x8

        x = self.avgpool(x)                     # 1x1
        x = self.flatten(x)                     # Flatten
        x  = self.fc(x)                         # Dense
        
        if print_sizes:
            print("\t In Model: output size", x.size())
            
        return x



def ResNet20(**kwargs):    
    return ResNet(name = 'ResNet20', depth = 20, **kwargs)

def ResNet32(**kwargs):
    return ResNet(name = 'ResNet32', depth = 32, **kwargs)

def ResNet44(**kwargs):
    return ResNet(name = 'ResNet44', depth = 44, **kwargs)

def ResNet56(**kwargs):
    return ResNet(name = 'ResNet56', depth = 56, **kwargs)

def ResNet110(**kwargs):
    return ResNet(name = 'ResNet110', depth = 110, **kwargs)



if __name__ == '__main__':

    import sys
    sys.path.append('..')
    from utils import count_parameters
    from beautifultable import BeautifulTable as BT

    resnet20 = ResNet20()
    resnet32 = ResNet32()
    resnet44 = ResNet44()
    resnet56 = ResNet56()
    resnet110 = ResNet110()
    
    table = BT()
    table.append_row(['Model', 'M. Paramars'])
    table.append_row(['ResNset20', count_parameters(resnet20)/1e6,])
    table.append_row(['ResNset32', count_parameters(resnet32)/1e6])
    table.append_row(['ResNset44', count_parameters(resnet44)/1e6])
    table.append_row(['ResNset56', count_parameters(resnet56)/1e6])
    table.append_row(['ResNset110', count_parameters(resnet110)/1e6])
    print(table)
        
    
    def test():
        net = ResNet56()
        y = net(Variable(torch.randn(1,3,32,32)))
        print(y.size())
    
    test()
    
    '''
    ResNets implemented on the paper <https://arxiv.org/pdf/1512.03385.pdf>
    
    +------------+-------------+
    |   Model    | M. Paramars |
    +------------+-------------+
    | ResNset20  |    0.272    |
    +------------+-------------+
    | ResNset32  |    0.467    |
    +------------+-------------+
    | ResNset44  |    0.661    |
    +------------+-------------+
    | ResNset56  |    0.856    |
    +------------+-------------+
    | ResNset110 |    1.731    |
    +------------+-------------+
    
    '''