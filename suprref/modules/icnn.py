import torch
from torch import nn


class ICNNModule(nn.Module):
    def __init__(self, num_classes, elements_length, non_elements_length, num_channels=200, num_hidden=2048, conv_kernel=21):
        super(ICNNModule, self).__init__()
        # The size of the convolution filter is set to 1×21×4, the number of filters used is set to 200, and the stride is set to 1.
        feature_map = non_elements_length - conv_kernel + 1
        self.conv = nn.Sequential(nn.Conv1d(in_channels=4, out_channels=num_channels, kernel_size=conv_kernel),
                                  nn.ReLU(), # This might be extra
                                  nn.MaxPool1d(feature_map))
        self.hidden = nn.Sequential(nn.Linear(elements_length + num_channels, num_hidden),
                                    nn.ReLU())
        self.out = nn.Linear(num_hidden, num_classes)
    
    def forward(self, x):
        non_elements = x[0]
        elements = x[1]
        convolved = self.conv(non_elements).squeeze(dim=-1)
        features = torch.cat((elements, convolved), dim=1)
        hidden = self.hidden(features)
        out = self.out(hidden).squeeze(dim=-1)
        return out