import torch
from torch import nn

#Under development. Right now it's the same as DPROM.
class OURModule(nn.Module):
    def __init__(self, num_classes, seqs_length=300, num_channels=64, num_hidden=128, dropout=0.5):
        super(OURModule, self).__init__()

        c1_kernel = 27
        p_kernel = 6
        self.c11 = nn.Sequential(nn.Conv1d(in_channels=4,
                                           out_channels=num_channels, kernel_size=c1_kernel),
                                 nn.ELU(),
                                 nn.MaxPool1d(p_kernel),
                                 nn.Dropout(dropout))
        c11_out = self.out_size(seqs_length, c1_kernel, p_kernel)

        c2_kernel = 14
        self.c12 = nn.Sequential(nn.Conv1d(in_channels=4,
                                           out_channels=num_channels, kernel_size=c2_kernel),
                                 nn.ELU(),
                                 nn.MaxPool1d(p_kernel),
                                 nn.Dropout(dropout))
        c12_out = self.out_size(seqs_length, c2_kernel, p_kernel)

        c3_kernel = 7
        self.c13 = nn.Sequential(nn.Conv1d(in_channels=4,
                                           out_channels=num_channels, kernel_size=c3_kernel),
                                 nn.ELU(),
                                 nn.MaxPool1d(p_kernel),
                                 nn.Dropout(dropout))
        c13_out = self.out_size(seqs_length, c3_kernel, p_kernel)

        concat_size = c11_out + c12_out + c13_out
        self.bilstm = nn.LSTM(
            num_channels, 32, 2, bias=True,
            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(64 * concat_size, num_hidden),
                                 nn.ELU(),
                                 nn.Dropout(dropout))
        self.fc2 = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        features = torch.cat((self.c11(x), self.c12(x), self.c13(x)), dim=2)
        rnn, _ = self.bilstm(features.transpose(1, 2))
        hidden = self.fc1(torch.flatten(rnn, 1))
        out = self.fc2(hidden).squeeze(dim=-1)
        return out

    @staticmethod
    def out_size(seqs_length, conv_kernel, pool_kernel):
        conv_shape = seqs_length - conv_kernel + 1
        pool_shape = ((conv_shape - 1* (pool_kernel - 1) - 1) // pool_kernel) + 1
        return pool_shape