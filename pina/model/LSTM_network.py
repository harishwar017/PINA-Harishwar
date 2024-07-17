"""Module for FeedForward model"""
import torch
import torch.nn as nn
from pina.label_tensor import LabelTensor


class LSTM(torch.nn.Module):
    
    def __init__(self, num_classes, input_variables, output_variables, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        if isinstance(input_variables, int):
            self.input_variables = None
            self.input_dimension = input_variables
        elif isinstance(input_variables, (tuple, list)):
            self.input_variables = input_variables
            self.input_dimension = len(input_variables)

        if isinstance(output_variables, int):
            self.output_variables = None
            self.output_dimension = output_variables
        elif isinstance(output_variables, (tuple, list)):
            self.output_variables = output_variables
            self.output_dimension = len(output_variables)
        
        self.num_classes = self.output_dimension
        self.num_layers = num_layers
        self.input_size = self.input_dimension
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out