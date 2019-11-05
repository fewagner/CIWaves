import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# diverges while training!!

def bdn(x,nr_mean,nr_std,ni_mean,ni_std): # short for Batch De Norm
    """
    Takes a normalized batch, together with mean and std so that it should be scaled back and returns the original sample.
    """
    
    x[:,:,:10001] = x[:,:,:10001]*nr_std + nr_mean
    x[:,:,10001:] = x[:,:,10001:]*ni_std + ni_mean
    
    return x

class CIWLSTM(nn.Module):
    def __init__(self, input_size, hidden_size_n, num_layers_n, seq_steps, device, 
                 hidden_size_psi, num_layers_psi, nr_mean, nr_std, ni_mean, ni_std):
        super(CIWLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size_n = hidden_size_n
        self.num_layers_n = num_layers_n
        self.hidden_size_psi = hidden_size_psi
        self.num_layers_psi = num_layers_psi
        self.seq_steps = seq_steps
        
        self.lstm_n = nn.LSTM(self.input_size +1, self.hidden_size_n, self.num_layers_n, batch_first=True)
        self.fc1_n = nn.Linear(self.hidden_size_n, self.input_size)
        self.fc2_n = nn.Linear(self.seq_steps * self.input_size +1, self.seq_steps * self.input_size)
        self.lstm_psi = nn.LSTM(2 * self.input_size + 1 , self.hidden_size_psi, self.num_layers_psi, batch_first=True)
        self.fc_psi = nn.Linear(self.hidden_size_psi, 2*self.input_size)
        
        self.device = device
        self.nr_mean = nr_mean
        self.nr_std = nr_std
        self.ni_mean = ni_mean
        self.ni_std = ni_std        
    
    
    def forward(self, knr):
        
        k_values = knr[:,0]
        nr = knr[:,1:]
        
        batchsize = nr.size(0)
        
        nr = nr.view(batchsize, self.seq_steps, self.input_size)
        
        k = torch.ones(batchsize, self.seq_steps, 1).to(self.device)
        
        for sample,k_val in zip(k,k_values):
            sample[:,:] = k_val
           
        # LSTM ni
        knr = torch.cat((k,nr), dim = 2).to(self.device)
        
        ni, _ = self.lstm_n(knr) # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        ni = ni.contiguous().view(batchsize * self.seq_steps , self.hidden_size_n)
        ni = F.relu(self.fc1_n(ni).view(batchsize , self.seq_steps * self.input_size))
        
        k_values = k_values.view(-1,1)
        ni = torch.cat((ni,k_values), dim = 1).to(self.device)
        ni = self.fc2_n(ni)
        
        inp = torch.cat((knr,ni.view(-1,self.seq_steps,self.input_size)), dim = 2).to(self.device)
        inp = bdn(inp, nr_mean, nr_std, ni_mean, ni_std)
        
        # LSTM Numerov       
        psi, _ =self.lstm_psi(inp) # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        psi = psi.contiguous().view(batchsize * self.seq_steps , self.hidden_size_psi) 
        psi = self.fc_psi(psi).view(batchsize , self.seq_steps * 2 * self.input_size) 
        
        psi_r = psi[:,:10000]
        psi_i = psi[:,10000:]
        
        # Intensity
        Int = psi_r **2 + psi_i **2
        
        return ni, psi_r, psi_i, Int
    