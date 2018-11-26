import torch
import torch.nn as nn
import torch.nn.functional as F

import global_variables

device = global_variables.device;

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,bi):
        super(EncoderRNN, self).__init__()
        self.bi=bi

        if self.bi:
            self.mul=2
        else:
            self.mul=1

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=self.bi)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, bs):
        return torch.zeros(self.mul, bs, self.hidden_size).to(device)



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, bi):
        super(DecoderRNN, self).__init__()
        self.bi = bi
        if self.bi:
            self.mul=2
        else:
            self.mul=1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True, bidirectional=self.bi)
        self.out = nn.Linear(self.mul * hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output).squeeze(dim=1))

        return output, hidden

    def initHidden(self):
        return torch.zeros(self.mul, bs, self.hidden_size).to(device)