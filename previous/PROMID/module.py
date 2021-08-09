import torch
from torch import nn

class PROMIDModule(nn.Module):
    def __init__(self, num_classes, seqs_length=300, num_channels=2, conv_kernel=15, pool_kernel=300):
        super(PROMIDModule, self).__init__()
        conv_shape = seqs_length - conv_kernel + 1
        pool_kernel = conv_shape
        pool_shape = ((conv_shape - 1* (pool_kernel - 1) - 1) // pool_kernel) + 1
        self.conv_pool = nn.Sequential(nn.Conv1d(in_channels=4, out_channels=num_channels, kernel_size=1),
                                       nn.AvgPool1d(pool_kernel))
        self.conv = nn.Conv1d(in_channels=4, out_channels=num_channels, kernel_size=conv_kernel)
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear((pool_shape + conv_shape) * num_channels, num_classes)

    def forward(self, x):
        features_conv = self.conv(x).squeeze(dim=-1)
        features_conv_pool = self.conv_pool(x)
        features = torch.cat([features_conv, features_conv_pool], dim=2).flatten(1)
        dropped = self.drop(features)
        out = self.out(dropped).squeeze(dim=-1)
        return out
