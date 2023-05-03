import torch.nn as nn

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # nn.MaxPool2d(2)
    )

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,64),
            conv_block(64,64),
            conv_block(64,64),
            conv_block(64,64)
        )
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        # x = nn.MaxPool2d(2)(x)
        return x.view(x.size(0),-1)
