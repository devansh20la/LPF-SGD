import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        # cxt.save_for_backward(input)
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        
        return output

    @staticmethod
    def backward(cxt, grad_output):
        # input, = cxt.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[torch.abs(input) > 1] = 0
        return grad_input

# aliases
binarize = BinarizeF.apply


class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output
        

class BinaryLinear(nn.Linear):

    def forward(self, input):
        binary_weight = binarize(self.weight)
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class BinaryConv2d(nn.Conv2d):

    def forward(self, input):
        bw = binarize(self.weight)
        return F.conv2d(input, bw, self.bias, self.stride,\
                               self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

# import torch
# import pdb
# import torch.nn as nn
# import math
# from torch.autograd import Variable
# from torch.autograd import Function

# import numpy as np

# def Binarize(tensor,quant_mode='det'):
#     if quant_mode=='det':
#         return tensor.sign()
#     else:
#         return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


# class BinaryLinear(nn.Linear):

#     def __init__(self, *kargs, **kwargs):
#         super(BinaryLinear, self).__init__(*kargs, **kwargs)

#     def forward(self, input):

#         if input.size(1) != 784:
#             input.data=Binarize(input.data)
#         if not hasattr(self.weight,'org'):
#             self.weight.org=self.weight.data.clone()
#         self.weight.data=Binarize(self.weight.org)
#         out = nn.functional.linear(input, self.weight)
#         if not self.bias is None:
#             self.bias.org=self.bias.data.clone()
#             out += self.bias.view(1, -1).expand_as(out)

#         return out


# class BinaryConv2d(nn.Conv2d):

#     def __init__(self, *kargs, **kwargs):
#         super(BinaryConv2d, self).__init__(*kargs, **kwargs)


#     def forward(self, input):
#         if input.size(1) != 3:
#             input.data = Binarize(input.data)
#         if not hasattr(self.weight,'org'):
#             self.weight.org=self.weight.data.clone()
#         self.weight.data=Binarize(self.weight.org)

#         out = nn.functional.conv2d(input, self.weight, None, self.stride,
#                                    self.padding, self.dilation, self.groups)

#         if not self.bias is None:
#             self.bias.org=self.bias.data.clone()
#             out += self.bias.view(1, -1, 1, 1).expand_as(out)

#         return out