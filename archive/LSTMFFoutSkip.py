import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#add some skip connections between lstm layers! Use selu activation

class LSTMFFout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_steps, device):
        super(LSTMFFout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_steps = seq_steps
        self.lstm1 = nn.LSTM(self.input_size +1, self.hidden_size, self.num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.input_size)
        self.fc2 = nn.Linear( self.seq_steps * self.input_size +1, self.seq_steps * self.input_size)
        self.device = device
    
    def forward(self, x):
        
        k_values = x[:,0]
        x = x[:,1:]
        
        #print(k_values)
        
        batchsize = x.size(0)
        
        x = x.view(batchsize, self.seq_steps, self.input_size)
        
        k = torch.ones(batchsize, self.seq_steps, 1).to(self.device)
        
        for sample,k_val in zip(k,k_values):
            sample[:,:] = k_val
            
        #print(k)
           
        inp = torch.cat((x,k), dim = 2).to(self.device)
        
        # Set initial hidden and cell states 
        #h0 = torch.zeros(self.num_layers, batchsize, self.hidden_size).to(self.device) 
        #c0 = torch.zeros(self.num_layers, batchsize, self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out0, _ = self.lstm1(inp)#, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #out, _ = self.lstm(x) # alternatively, without setting h0 and c0
        out1, _ = self.lstm2(out0)
        out1 = out1 + out0
        out2, _ = self.lstm3(out1)
        out2 = out2 + out1
        out2 = out2.contiguous().view(batchsize * self.seq_steps , self.hidden_size)
        
        out2 = F.selu(self.fc1(out2).view(batchsize , self.seq_steps * self.input_size))
        
        k_values = k_values.view(-1,1)
        
        #print(out.size())
        #print(k_values.size())        
        out2 = torch.cat((out2,k_values), dim = 1).to(self.device)
        out2 = self.fc2(out2)
        
        return out2