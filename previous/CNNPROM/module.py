import torch
from torch import nn

class CNNPROMModule(nn.Module):
    def __init__(self, num_classes, seqs_length=300, num_channels=200, num_hidden=128, conv_kernel=21, pool_kernel=4):
        super(CNNPROMModule, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv1d(in_channels=4,
                                           out_channels=num_channels, kernel_size=conv_kernel),
                                 nn.ReLU(),
                                 nn.MaxPool1d(pool_kernel))
        conv_shape = seqs_length - conv_kernel + 1
        pool_shape = ((conv_shape - 1* (pool_kernel - 1) - 1) // pool_kernel) + 1
        self.hidden = nn.Sequential(nn.Linear(num_channels*pool_shape, num_hidden),
                                 nn.ReLU())
        self.out = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        features = self.conv(x).squeeze(dim=-1)
        features = features.reshape(features.shape[0], -1)
        hidden = self.hidden(features)
        out = self.out(hidden).squeeze(dim=-1)
        return out
