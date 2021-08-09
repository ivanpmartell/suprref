import torch
from torch import nn


class ICNNModule(nn.Module):
    def __init__(self, num_classes, elements_length, non_elements_length, num_channels=200, num_hidden=2048, conv_kernel=21):
        super(ICNNModule, self).__init__()

        feature_map = non_elements_length - conv_kernel + 1
        self.conv = nn.Sequential(nn.Conv1d(in_channels=4, out_channels=num_channels, kernel_size=conv_kernel),
                                  nn.ReLU(),
                                  nn.MaxPool1d(feature_map))
        self.hidden = nn.Sequential(nn.Linear(elements_length + num_channels, num_hidden),
                                    nn.ReLU())
        self.out = nn.Linear(num_hidden, num_classes)
    
    def forward(self, x):
        if len(x[1].shape) == 1:
            elements = x[0][0]
            non_elements = x[0][1]
        else:
            elements = x[0]
            non_elements = x[1]
        convolved = self.conv(non_elements).squeeze(dim=-1)
        features = torch.cat((elements, convolved), dim=1)
        hidden = self.hidden(features)
        out = self.out(hidden).squeeze(dim=-1)
        return out