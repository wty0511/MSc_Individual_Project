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
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('total_trainable_params:', total_trainable_params)
    def forward(self, src):
        src = src.permute(0,2,1).contiguous()
        # print('src shape ',src.shape)
        output = self.transformer_encoder(src)
        output = output.permute(0,2,1).contiguous()
        output = self.conv(output)
        return output


# class Transformer(nn.Module):
#     def __init__(self, input_dim, num_heads, num_layers):
#         super(Transformer, self).__init__()

#         # Transformer Encoder
#         encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))   # CLS token
#         # Fully Connected Layer for Classification
#         total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

#         print('total_trainable_params:', total_trainable_params)
#     def forward(self, src):
#         src = src.permute(0,2,1).contiguous()
#         # print('src shape ',src.shape)
#         # print('cls shape ',self.cls_token.shape)

#         cls_tokens = self.cls_token.expand(src.size(0), -1, -1)
#         src = torch.cat([cls_tokens, src], dim=1)
#         # print('src shape ',src.shape)
#         output = self.transformer_encoder(src)
#         output = output.permute(0,2,1).contiguous()
#         output = output[:, 0, :]
#         return output

