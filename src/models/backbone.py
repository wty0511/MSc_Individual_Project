# This code is originally provided by https://github.com/facebookresearch/low-shot-shrink-hallucinate
# And is further modified by https://github.com/wyharveychen/CloserLookFewShot
# I have modified it to fit my project
import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn import init

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        # the weitht of the conv layer is initialized by the fan-in method, i.e. the variance of the weight is 2/(k*k*cin), where k is the kernel size, and cin is the number of input channels
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple Conv Block
class ConvBlock(nn.Module):

    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim

        self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
        self.BN     = BatchNorm2d_fw(outdim)
        self.relu   = nn.ReLU(inplace=True)
        # self.relu   = nn.LeakyReLU(0.1, inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        # self.parametrized_layers = [self.C,  self.relu]
        
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out



class ConvNetClassifierfw(nn.Module):
    def __init__(self, depth = 3):
        super(ConvNetClassifierfw,self).__init__()
        self.conv = ConvNetfw_large(depth = 4)
        self.fc = Linear_fw(512, 2)
        # init.xavier_uniform_(self.fc.weight)

        # print(self.fc.weight.data)
        # print(self.fc.bias.data)
        self.fc.bias.data.fill_(0)
        # print('weight',self.fc.bias.data)
        
    def forward(self,x):
        out = self.conv(x)
        # print('out',out.shape)
        out = self.fc(out)
        return out
    
    


class ConvNetClassifierSmallfw(nn.Module):
    def __init__(self, depth = 3):
        super(ConvNetClassifierSmallfw,self).__init__()
        self.conv = ConvNetfw_small(depth = 3)
        self.fc = Linear_fw(512, 2)
        # init.xavier_uniform_(self.fc.weight)

        # print(self.fc.weight.data)
        # print(self.fc.bias.data)
        self.fc.bias.data.fill_(0)
        # print('weight',self.fc.bias.data)
        
    def forward(self,x):
        out = self.conv(x)
        # print('out',out.shape)
        out = self.fc(out)
        return out
    


class ConvNetfw_large(nn.Module):
    def __init__(self, depth = 4):
        super(ConvNetfw_large,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 128
            outdim = 128
            B = ConvBlock(indim, outdim, pool = ( i < 3 ) ) #only pooling for fist 4 layers
            trunk.append(B)
        self.trunk = nn.Sequential(*trunk)
        self.avgpool = nn.AdaptiveAvgPool2d((4,1))
    def forward(self,x):

        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        out = self.trunk(x)
        out = self.avgpool(out)
        out = out.view(x.size(0),-1)
        return out
    


class ConvNetfw_small(nn.Module):
    def __init__(self, depth = 3):
        super(ConvNetfw_small,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i < 3 ) ) #only pooling for fist 4 layers
            trunk.append(B)
        self.trunk = nn.Sequential(*trunk)
        self.avgpool = nn.AdaptiveAvgPool2d((8,1))
    def forward(self,x):

        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        out = self.trunk(x)
        out = self.avgpool(out)
        out = out.view(x.size(0),-1)
        return out
    

    
# class ConvNetfw(nn.Module):
#     def __init__(self, depth = 3):
#         super(ConvNetfw,self).__init__()
#         trunk = []
#         for i in range(depth):
#             indim = 1 if i == 0 else 64
#             outdim = 64
#             B = ConvBlock(indim, outdim, pool = ( i < 3 ) ) #only pooling for fist 4 layers
#             trunk.append(B)
#         self.trunk = nn.Sequential(*trunk)
#         self.avgpool = nn.AdaptiveAvgPool2d((8,1))
#     def forward(self,x):

#         (num_samples,seq_len,mel_bins) = x.shape
#         x = x.view(-1,1,seq_len,mel_bins)
#         out = self.trunk(x)
#         out = self.avgpool(out)
#         out = out.view(x.size(0),-1)
#         return out
    
class SNNfw(nn.Module):
    def __init__(self, depth = 3):
        super(SNNfw,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i < 3 ) ) #only pooling for fist 3 layers
            trunk.append(B)
        self.trunk = nn.Sequential(*trunk)
        self.avgpool = nn.AdaptiveAvgPool2d((8,1))
        self.fc1 = Linear_fw(512, 256)
        self.fc2 = Linear_fw(256, 1)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.activation = nn.Sigmoid()
        

    
    def forward(self,*args):
        if len(args) == 1:
            x = args[0]
            (num_samples,seq_len,mel_bins) = x.shape
            x = x.view(-1,1,seq_len,mel_bins)
            out = self.trunk(x)
            out = self.avgpool(out)
            out = out.view(x.size(0),-1)
            return out
        if len(args) == 2:
            x1,x2 = args
            (num_samples,seq_len,mel_bins) = x1.shape
            
            x1 = x1.view(-1,1,seq_len,mel_bins)
            x2 = x2.view(-1,1,seq_len,mel_bins)
            
            out1 = self.trunk(x1)
            out1 = self.avgpool(out1)

            out2 = self.trunk(x2)
            out2 = self.avgpool(out2)
            
            out = torch.abs(out1 - out2)
            out = out.view(x1.size(0),-1)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.activation(out)

            return out

