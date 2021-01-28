import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BinaryConv2d, BinaryLinear, BinaryTanh

# BinaryTanh = nn.Hardtanh

class BinaryLeNet(nn.Module):
    def __init__(self):
        super(BinaryLeNet, self).__init__()
        self.conv1 = nn.Sequential(BinaryConv2d(1, 20, 5, 1),
                            nn.BatchNorm2d(20),
                            nn.MaxPool2d(2),
                            BinaryTanh())
        self.conv2 = nn.Sequential(BinaryConv2d(20, 50, 5, 1),
                            nn.BatchNorm2d(50),
                            nn.MaxPool2d(2),
                            BinaryTanh())
        
        self.fc1 = nn.Sequential(BinaryLinear(800, 500),
                            nn.BatchNorm1d(500),
                            BinaryTanh())
        
        self.fc2 = nn.Sequential(BinaryLinear(500, 10),
                            nn.BatchNorm1d(10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 800)
        x = self.fc1(x)
        x = self.fc2(x)
        return x