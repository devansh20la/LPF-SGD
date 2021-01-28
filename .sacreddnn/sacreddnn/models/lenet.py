import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    ''' Init Function Input:
            First row: Input parameters
            Second & Third row: Convolution parameters
            Fourth row: MaxPool parameters
            Fifth row: FC and output parameters
    '''
    def __init__(self, dim=32, in_channels=1,
                 out_channels_1=6, out_channels_2=16,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize1=120, fcsize2=84, nclasses=10, activation=nn.ReLU(), dropout=0.):

        super().__init__()

        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        self.max_pool = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        self.fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize1)
        self.fc2 = nn.Linear(fcsize1, fcsize2)
        self.fc3 = nn.Linear(fcsize2, nclasses)
        #self.relu = nn.ReLU()
        self.relu = activation
        
        # dropout
        #self.dropout2d = nn.Dropout2d(p=dropout)
        self.dropout1d = nn.Dropout(p=dropout)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout1d(x)      # dropout
        x = self.relu(self.fc2(x))
        x = self.dropout1d(x)      # dropout
        x = self.fc3(x)
        return x

    
class LeNet5_bn(nn.Module):
    ''' Init Function Input:
            First row: Input parameters
            Second & Third row: Convolution parameters
            Fourth row: MaxPool parameters
            Fifth row: FC and output parameters
    '''
    def __init__(self, dim=32, in_channels=1,
                 out_channels_1=6, out_channels_2=16,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize1=120, fcsize2=84, nclasses=10, activation=nn.ReLU(), dropout=0.):

        super().__init__()

        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        self.max_pool = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)

        self.conv_bn1 = nn.BatchNorm2d(out_channels_1)
        self.conv_bn2 = nn.BatchNorm2d(out_channels_2)
        
        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        self.fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize1)
        self.fc2 = nn.Linear(fcsize1, fcsize2)
        self.fc3 = nn.Linear(fcsize2, nclasses)
        
        self.fc_bn1 = nn.BatchNorm1d(fcsize1)
        self.fc_bn2 = nn.BatchNorm1d(fcsize2)

        #self.relu = nn.ReLU()
        self.relu = activation
        
        # dropout
        #self.dropout2d = nn.Dropout2d(p=dropout)
        self.dropout1d = nn.Dropout(p=dropout)
        

    def forward(self, x):
        x = self.relu(self.conv_bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.relu(self.conv_bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout1d(x)      # dropout
        x = self.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout1d(x)      # dropout
        x = self.fc3(x)
        return x
    
    
class LeNet(nn.Module):
    ''' Init Function Input:
            First row: Input parameters
            Second & Third row: Convolution parameters
            Fourth row: MaxPool parameters
            Fifth row: FC and output parameters
    '''
    def __init__(self, dim=32, in_channels=1,
                 out_channels_1=20, out_channels_2=50,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize=500, nclasses=10, activation=nn.ReLU(), dropout=0.):

        super().__init__()

        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        self.max_pool = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        self.fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize)
        self.fc2 = nn.Linear(fcsize, nclasses)
        #self.relu = nn.ReLU()
        self.relu = activation
        
        self.dropout1d = nn.Dropout(p=dropout)

    def forward(self, x):
        # print(type(self.relu), type(self.conv1))
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout1d(x)      # dropout
        x = self.fc2(x)
        return x

    
class LeNet_bn(nn.Module):
    ''' Init Function Input:
            First row: Input parameters
            Second & Third row: Convolution parameters
            Fourth row: MaxPool parameters
            Fifth row: FC and output parameters
    '''
    def __init__(self, dim=32, in_channels=1,
                 out_channels_1=20, out_channels_2=50,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize=500, nclasses=10, activation=nn.ReLU(), dropout=0.):

        super().__init__()

        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        self.max_pool = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)

        self.conv_bn1 = nn.BatchNorm2d(out_channels_1)
        self.conv_bn2 = nn.BatchNorm2d(out_channels_2)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        self.fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize)
        self.fc2 = nn.Linear(fcsize, nclasses)
        
        self.fc_bn1 = nn.BatchNorm1d(fcsize)

        #self.relu = nn.ReLU()
        self.relu = activation

        self.dropout1d = nn.Dropout(p=dropout)


    def forward(self, x):
        x = self.relu(self.conv_bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.relu(self.conv_bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout1d(x)      # dropout
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    m = LeNet(dim=32, in_channels=3, nclasses=10, activation=nn.ReLU(), dropout=0.1)
    print(m)


