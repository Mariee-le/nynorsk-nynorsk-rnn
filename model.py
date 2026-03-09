import torch
import torch.nn as nn



class CharRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
    
        super().__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
   
        out, hidden = self.rnn(x)
        out = self.fc(hidden[0])
        out = self.softmax(out)

        return out
