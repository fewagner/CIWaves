import torch.nn as nn
import torch.nn.functional as F
import torch


class ODESolver(nn.Module):

    def __init__(self, steps, n_hidden): #input k,h,nr,ni: 1,1,steps+1,steps
        super(ODESolver, self).__init__()
        self.hidden1 = nn.Linear(2*steps + 3, n_hidden)  # hidden layer
        self.hidden2 = nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.hidden3 = nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.predict = nn.Linear(n_hidden, 1)  # output layer

    def forward(self, x):
        x = F.selu(self.hidden1(x))  # activation function for hidden layer
        x = F.selu(self.hidden2(x))
        x = F.selu(self.hidden3(x))
        x = self.predict(x)  # linear output
        return x
