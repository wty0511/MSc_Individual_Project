import torch.nn as nn
import torch
def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # nn.LeakyReLU(),
        # nn.SELU(),
        nn.MaxPool2d(2)
    )

def conv_block2(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d((2,1))
        # nn.LeakyReLU(),
        # nn.SELU(),
    )

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet,self).__init__()
#         self.encoder = nn.Sequential(
#             conv_block(1,64),
#             conv_block(64,64),
#             conv_block(64,64)       
#             # conv_block(64,64)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((8,1))
#         total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

#         print('total_trainable_params:', total_trainable_params)
#         # self.temperature_param = nn.Parameter(torch.tensor(1.0))
#     def forward(self,x):
#         (num_samples,seq_len,mel_bins) = x.shape
#         x = x.view(-1,1,seq_len,mel_bins)
#         x = self.encoder(x)
#         # print('x_shape ',x.shape)
#         x = self.avgpool(x)
#         # x = nn.MaxPool2d(2)(x)
#         x = x.view(x.size(0),-1)
#         return x



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.encoder = nn.Sequential(
            conv_block2(1,64),
            conv_block2(64,64),
            conv_block2(64,64),
            conv_block2(64,64)
        )
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(1,1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1,1))
        self.activ = torch.nn.Sigmoid ()
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('total_trainable_params:', total_trainable_params)
        # self.temperature_param = nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape

        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        x1 = self.activ(x1)
        sums = torch.sum(x1, dim=-1, keepdim=True)
        x1 = x1 / sums
        x2 = x1 * x2
        x2 = x2.view(x2.size(0),-1, x2.size(-1))
        x2 = torch.sum(x2, dim=-1)
        # print(x2.shape)
        return x2
    

class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,64),
            conv_block(64,64),
            conv_block(64,64),
            # conv_block(64,64)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8,1))
        self.fc = nn.Conv2d(64, 2, kernel_size=(8,1))
        
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape

        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        # print('x_shape ',x.shape)
        x = self.avgpool(x)
        # x = nn.MaxPool2d(2)(x)
        x = self.fc(x)
        x = x.view(x.size(0),-1)
        
        return x


class ConvSNN(nn.Module):
    def __init__(self):
        super(ConvSNN,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,64),
            conv_block(64,64),
            conv_block(64,64),
            # conv_block(64,64)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8,1))
        self.fc1 = nn.Conv2d(64, 64, kernel_size=(8,1))
        self.fc2 = nn.Conv2d(64, 1, kernel_size=(1,1))
        self.activ = torch.nn.Sigmoid ()
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('total_trainable_params:', total_trainable_params)
    def forward(self,x1, x2):
        (num_samples,seq_len,mel_bins) = x1.shape

        x1 = x1.view(-1,1,seq_len,mel_bins)
        x2 = x2.view(-1,1,seq_len,mel_bins)
        
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        # print('x_shape ',x.shape)
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)
        # x1 = x1.view(x1.size(0),-1)
        # x2 = x2.view(x2.size(0),-1)
        x = torch.abs(x1 - x2)
        
        # x = nn.MaxPool2d(2)(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        x = x.view(x.size(0))
        x = self.activ(x)
        
        return x