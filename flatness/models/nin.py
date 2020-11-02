import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_layer(object):
    """docstring for conv_layer"""
    def __init__(self, args, **kwargs):
        super(conv_layer, self).__init__()
        self.batchnorm = args.batchnorm
        if self.batchnorm:
            self.conv = nn.Conv2d(**kwargs)
            self.batchnorm = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv2d(**kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        return F.relu(x)


class nin(nn.Module):
    def __init__(self, args):
        super(nin, self).__init__()

        self.layer1 = conv_layer(args, in_channels=192, out_channels=160, kernel_size=1, stride=1, padding=0)

                # nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                # nn.Dropout(0.5),

                # nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(inplace=True),
                # nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                # nn.Dropout(0.5),

                # nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                # nn.ReLU(inplace=True),
                # nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                # )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), 10)
        return x


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from args import get_args

    args = get_args(["--exp_num", "0"])
    model = nin(args)

    model = nin()