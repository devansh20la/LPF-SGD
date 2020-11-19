import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_layer(nn.Module):
    """docstring for conv_layer"""
    def __init__(self, args, in_channels, out_channels, kernel_size, stride, padding):
        super(conv_layer, self).__init__()
        self.batchnorm = args.batchnorm
        if self.batchnorm:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.batchnorm = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        return x


class nin(nn.Module):
    def __init__(self, args):
        super(nin, self).__init__()
        self.skip = args.skip
        self.batchnorm = args.batchnorm
        self.num_classes = args.num_classes
        self.layer1 = nn.Sequential(
            conv_layer(args, in_channels=3, out_channels=192, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
            conv_layer(args, in_channels=192, out_channels=160, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
            conv_layer(args, in_channels=160, out_channels=192, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer2 = nn.Sequential(
            conv_layer(args, in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
            conv_layer(args, in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
            conv_layer(args, in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)
        )

        self.layer2_ex = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.layer3 = nn.Sequential(
            conv_layer(args, in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
            conv_layer(args, in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
            conv_layer(args, in_channels=192, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        )

        if self.skip:
            self.downsample = nn.Sequential()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        out = self.layer2(x)
        if self.skip:
            out += self.downsample(x)
        out = self.layer2_ex(out)
        out = self.layer3(out)
        return out.view(-1, self.num_classes)


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from args import get_args

    args = get_args(["--exp_num", "0"])
    args.skip = True
    args.batchnorm = False

    model = nin(args)
    x = torch.randn(1, 3, 32, 32)
    print(model, model(x).shape)

