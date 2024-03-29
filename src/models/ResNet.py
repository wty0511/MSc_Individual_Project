# This code is modified from Deep Learning Courcework
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    # 来自于 DL CW1
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False, groups=1),
                nn.BatchNorm2d(output_channels)
            )
    
    def forward(self, x):
        out= self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        # print(self.training)
        out = F.dropout(out, p=0.5, training=self.training)
        return out 

class ResNet(nn.Module):
    
    def __init__(self,input_channels, block):
        super(ResNet, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(1, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, 2, stride=2)
        self.layer2 = self._make_layer(block, 64, 2, stride=2)
        self.layer3 = self._make_layer(block, 64, 2, stride=2)
        # self.layer4 = self._make_layer(block, 64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((8,1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total_trainable_params = sum(p.numel() for p in self.layer2.parameters() if p.requires_grad)
        print('total_trainable_params:', total_trainable_params)
    def _make_layer(self, block, output_channels, n_layers,stride=1):
        layers = []
        layers.append(block(self.input_channels, output_channels, stride))
        self.input_channels = output_channels
        for _ in range(1, n_layers):
            layers.append(block(self.input_channels, output_channels, 1))

        return nn.Sequential(*layers)
    def forward(self, x):
        # print(x.shape)
        x = x.to(self.device)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        # x = self.layer4(x)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print('~~~~~')
        return x


# class ResBlock(nn.Module):
#     # 来自于 DL CW1
#     def __init__(self, input_channels, output_channels, stride=1):
#         super(ResBlock, self).__init__()
        
#         self.conv1_d = nn.Conv2d(input_channels, 1, kernel_size=3, stride=stride, padding=1, bias=False, groups=1)
        
#         self.conv1_p = nn.Conv2d(1, output_channels, kernel_size=1, bias=False, groups=1)
        
#         self.bn1_d = nn.BatchNorm2d(1)
#         self.bn1_p = nn.BatchNorm2d(output_channels)
        
        
#         self.conv2_d = nn.Conv2d(output_channels, 1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
#         self.conv2_p = nn.Conv2d(1, output_channels, kernel_size=1, bias=False, groups=1)
        
#         self.bn2_d = nn.BatchNorm2d(1)
#         self.bn2_p = nn.BatchNorm2d(output_channels)
        
#         self.relu = nn.LeakyReLU(inplace=True)
        
#         self.shortcut = nn.Sequential()
        
#         if stride != 1 or input_channels != output_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False, groups=1),
#                 nn.BatchNorm2d(output_channels)
#             )
    
#     def forward(self, x):
#         out= self.conv1_d(x)
#         out = self.bn1_d(out)
#         out = self.relu(out)
#         out = self.conv1_p(out)
#         out = self.bn1_p(out)
#         out = self.relu(out)
#         out = self.conv2_d(out)
#         out = self.bn2_d(out)
#         out = self.relu(out)
#         out = self.conv2_p(out)
#         out = self.bn2_p(out)
#         out += self.shortcut(x)
#         out = self.relu(out)
#         # print(self.training)
#         out = F.dropout(out, p=0.5, training=self.training)
#         return out 

# class ResNet(nn.Module):
    
#     def __init__(self,input_channels, block):
#         super(ResNet, self).__init__()
#         self.input_channels = input_channels
#         self.conv1 = nn.Conv2d(1, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(input_channels)
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.layer1 = self._make_layer(block, 64, 2, stride=1)
#         self.layer2 = self._make_layer(block, 128, 2, stride=2)
#         self.layer3 = self._make_layer(block, 256, 2, stride=2)
#         self.layer4 = self._make_layer(block, 64, 2, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((8,1))
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print('total_trainable_params:', total_trainable_params)
#     def _make_layer(self, block, output_channels, n_layers,stride=1):
#         layers = []
#         layers.append(block(self.input_channels, output_channels, stride))
#         self.input_channels = output_channels
#         for _ in range(1, n_layers):
#             layers.append(block(self.input_channels, output_channels, 1))

#         return nn.Sequential(*layers)
#     def forward(self, x):
#         # print(x.shape)
#         x = x.to(self.device)
#         x = x.unsqueeze(1)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.layer1(x)
        
#         x = self.layer2(x)
#         x = self.layer3(x)
#         # print(x.shape)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         # print(x.shape)
#         x = x.view(x.size(0), -1)
#         # print('~~~~~')
#         return x