import torch
from torch import nn
import torch.nn.functional as F
from .helper import Helper

class CNNModule(nn.Module):
    def __init__(self, num_classes, seqs_length=None, num_filters=1, kernel_size=15, padding=0, stride=1):
        super(CNNModule, self).__init__()
        helper_class = Helper()
        
        self.conv = nn.Conv1d(in_channels=len(helper_class.get_DNA_dict()), out_channels=num_filters,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        conv_shape = Helper.conv1d_out_shape(seqs_length, kernel_size, padding, stride)
        self.out = nn.Linear(num_filters*conv_shape, num_classes)

    def forward(self, x):
        features = F.elu(self.conv(x)).flatten(1)
        out = self.out(features)
        return out

class JASPARModule(nn.Module):
    filters: torch.tensor
    filter_names: list
    padding: torch.tensor
    stride: torch.tensor
    def __init__(self, num_classes, seqs_length=None, matrices=None, matrix_type='ppm', padding=0, stride=1):
        super(JASPARModule, self).__init__()
        max_filter_len, pms = Helper.get_filters(matrices, matrix_type)
        padded = []
        names = []
        for pm in pms:
            names.append(pm[0])
            pm = pm[1]
            dif = max_filter_len - pm.shape[1]
            if(dif > 0):
                padded.append(nn.ConstantPad1d((0,dif), 0.)(torch.FloatTensor(pm)))
            else:
                padded.append(torch.FloatTensor(pm))
        self.filter_names = names
        padding = max_filter_len - 1
        self.filters = torch.stack(padded, axis=0)
        conv_shape = Helper.conv1d_out_shape(seqs_length, max_filter_len, padding, stride)
        self.out = nn.Linear(len(matrices)*conv_shape, num_classes)
        self.padding = padding
        self.stride = stride
        
    
    def forward(self, x):
        conv = F.conv1d(x, self.filters, padding=self.padding, stride=self.stride)
        features = F.elu(conv).flatten(1)
        out = self.out(features)
        return out

class LSTMModule(nn.Module):
    def __init__(self, num_classes, seqs_length=None, hidden_size=32):
        super(LSTMModule, self).__init__()
        helper_class = Helper()

        self.lstm = nn.LSTM(len(helper_class.get_DNA_dict()), num_layers=2, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.help = nn.Linear(hidden_size * 2, len(helper_class.get_DNA_dict()))
        self.out = nn.Linear(seqs_length * len(helper_class.get_DNA_dict()), num_classes)
    
    def forward(self, x):
        rnn, _ = self.lstm(x.transpose(1, 2))
        features = F.sigmoid(self.help(rnn))
        focus = features * x.transpose(1,2)
        out = self.out(focus.reshape(focus.size(0),-1))
        #out = self.out(hidden[0].squeeze(0))
        return out

class CNNLSTMModule(nn.Module):
    def __init__(self, num_classes, seqs_length=None, hidden_size=32, num_filters=1, kernel_size=26, padding=0, stride=1):
        super(CNNLSTMModule, self).__init__()
        helper_class = Helper()

        self.conv = nn.Conv1d(in_channels=len(helper_class.get_DNA_dict()), out_channels=num_filters,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        conv_shape = Helper.conv1d_out_shape(seqs_length, kernel_size, padding, stride)

        self.lstm = nn.LSTM(len(helper_class.get_DNA_dict()), hidden_size=hidden_size, batch_first=True)
        self.help = nn.Linear(hidden_size, len(helper_class.get_DNA_dict()))
        self.out = nn.Linear(seqs_length * len(helper_class.get_DNA_dict()), num_classes)
    
    def forward(self, x):
        convolved = F.elu(self.conv(x)).flatten(1)
        #TODO: use convolved
        rnn, _ = self.lstm(x.transpose(1, 2))
        features = F.sigmoid(self.help(rnn))
        focus = features * x.transpose(1,2)
        out = self.out(focus.reshape(focus.size(0),-1))
        #out = self.out(hidden[0].squeeze(0))
        return out