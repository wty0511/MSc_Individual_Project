import torch
from torch import nn
from src.models.ConvNet import *
class Transformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(Transformer, self).__init__()

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.conv = ConvNet()
        # Fully Connected Layer for Classification
        
    def forward(self, src):

        src = src.permute(0,2,1)
        # print('src shape ',src.shape)
        embedded = src
        output = self.transformer_encoder(embedded)
        output = output.permute(0,2,1)
        output = self.conv(output)
        return output
