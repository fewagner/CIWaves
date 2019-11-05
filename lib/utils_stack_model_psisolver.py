import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import visdom

from lib.utils_train import calc_psi_Int
from lib.utils_evaluation import DeNorm


def train_lstm(criterion,
               model_optimizer,
               n_i, 
               Int,
               ni_pred, 
               psir_pred, 
               psii_pred):

    loss_ni = criterion(n_i, ni_pred)
    
    Int_pred = psir_pred**2 + psii_pred**2
    loss_Int = criterion(Int_pred, Int) # Int is just ones
    
    loss_model = loss_ni + loss_Int
    
    model_optimizer.zero_grad()
    loss_model.backward(retain_graph=True)
    model_optimizer.step()
    
    return loss_ni.item(), loss_Int.item()


def train_psisolver(criterion,
                 psisolver_optimizer,
                 psir,
                 psii,
                 psir_pred,
                 psii_pred):
    loss_psir = criterion(psir,psir_pred)
    loss_psii = criterion(psii,psii_pred)
    loss_psi = loss_psir + loss_psii
    
    psisolver_optimizer.zero_grad()
    loss_psi.backward()
    psisolver_optimizer.step()    

    return loss_psi.item()


def get_psi(grid, stepsize, knr, ni, Amp=1):

    psir = torch.zeros(0)
    psii = torch.zeros(0)
    Int = torch.zeros(0)

    for one_knr, one_ni in zip(knr, ni):
        one_psir, one_psii, one_Int = calc_psi_Int(grid, stepsize,
                n = one_knr[1:].detach().cpu().numpy() + \
                1j*one_ni.detach().cpu().numpy(),
                Amp = 1,
                k = one_knr[0].detach().cpu().numpy())
        psir = torch.cat(
            (psir, torch.from_numpy(one_psir).float().view(1, -1)), dim=0)
        psii = torch.cat(
            (psii, torch.from_numpy(one_psii).float().view(1, -1)), dim=0)
        Int = torch.cat(
            (Int, torch.from_numpy(one_Int).float().view(1, -1)), dim=0)

    return psir, psii, Int


def stack_init_visdom():
    
    vis = visdom.Visdom()
    
    train_ni_loss = 1
    validation_ni_loss = 1
    train_psi_loss = 1
    validation_psi_loss = 1
    train_Int_loss = 1
    validation_Int_loss = 1
    
    visdom_model = vis.line(
        X=np.column_stack((0, 0, 0, 0)),
        Y=np.column_stack((train_ni_loss,
                           train_Int_loss,
                           validation_ni_loss,
                           validation_Int_loss)),
        opts={
            'title': 'Model Loss',
            'legend': ['ni_train', 'Int_train', 'ni_validation', 'Int_validation']}
    )
    
    visdom_psisolver = vis.line(
        X=np.column_stack((0, 0)),
        Y=np.column_stack((train_psi_loss,
                           validation_psi_loss)),
        opts={
            'title': 'Psisolver Loss',
            'legend': ['train', 'validation']}
    )
    
    return vis, visdom_model, visdom_psisolver


def stack_visdom_point(vis,  
                       current_epoch,
                       visdom_model, 
                       visdom_psisolver, 
                       validation_loader,
                       criterion, 
                       train_ni_loss, 
                       train_psi_loss,
                       train_Int_loss, 
                       grid, 
                       stepsize,
                       device,
                       model,
                       psisolver,
                        nr_mean,
                        nr_std,
                        ni_mean,
                        ni_std):

    validation_ni_loss = 0
    validation_psi_loss = 0
    validation_Int_loss = 0

    for i, batch in enumerate(validation_loader):

        if i % 50 == 0:
            print('validation batch {}/{}'.format(i, len(validation_loader)))

        # unravel data
        k_n_r = batch['k_n_r'].to(device)
        n_i = batch['n_i'].to(device)

        # make predictions and calc psi and Int
        ni_pred = model(k_n_r)

        psir_pred, psii_pred = psisolver(DeNorm(k_n_r, nr_mean, nr_std),
                                       DeNorm(ni_pred, ni_mean, ni_std))

        psir, psii, _ = get_psi(grid,
                             stepsize,
                             DeNorm(k_n_r, nr_mean, nr_std),
                             DeNorm(ni_pred, ni_mean, ni_std),
                             Amp=1)

        psir, psii = psir.to(device), psii.to(device)

        Int_pred = psir_pred**2 + psii_pred**2

        # sum up losses
        validation_ni_loss += \
            criterion(n_i, ni_pred).item()/len(validation_loader)

        validation_psi_loss += \
            criterion(psir, psir_pred).item()/len(validation_loader) + \
            criterion(psii, psii_pred).item()/len(validation_loader)

        validation_Int_loss += \
            criterion(Int_pred, torch.ones(Int_pred.shape).to(device))\
            .item()/len(validation_loader)

        # add visdom point
        vis.line(X=np.column_stack((current_epoch + 1, current_epoch + 1,
                                    current_epoch + 1, current_epoch + 1)),
                 Y=np.column_stack((train_ni_loss, train_Int_loss,
                                    validation_ni_loss, validation_Int_loss)),
                 win=visdom_model,
                 update='append')

        vis.line(X=np.column_stack((current_epoch + 1, current_epoch + 1)),
                 Y=np.column_stack((train_psi_loss, validation_psi_loss)),
                 win=visdom_psisolver,
                 update='append')

    return validation_ni_loss, \
           validation_psi_loss, \
           validation_Int_loss


class PsiSolver(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_steps, device):
        super(PsiSolver, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # size of ni input -
        #                                 actual input size is 2* +1
        self.num_layers = num_layers
        self.seq_steps = seq_steps

        self.lstm = nn.LSTM(2 * self.input_size + 1,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 2 * self.input_size)
        self.fc2 = nn.Linear(self.seq_steps * 2 * self.input_size + 1,
                               self.seq_steps * 2 * self.input_size)

        self.device = device

    def forward(self, knr, ni):

        k_values = knr[:, 0]  # all k of batch, size (bs,1)
        nr = knr[:, 1:]

        batchsize = nr.size(0)

        nr = nr.view(batchsize, self.seq_steps, self.input_size)
        ni = ni.view(batchsize, self.seq_steps, self.input_size)

        k = torch.ones(batchsize, self.seq_steps, 1).to(self.device)

        for sample, k_val in zip(k, k_values):
            sample[:, :] = k_val

        # LSTM
        inp = torch.cat((k, nr, ni),
                        dim=2).to(self.device)  # size (bs, seql, 2*inps+1)

        outp, _ = self.lstm(inp)  # out: tensor of shape (batch_size, seq_steps, hidden_size)

        outp = outp.contiguous().view(batchsize * self.seq_steps,
                                      self.hidden_size)
        outp = F.relu(
            self.fc1(outp).view(batchsize,
                                self.seq_steps * 2 * self.input_size))

        k_values = k_values.view(-1,1)
        outp = torch.cat((outp, k_values), dim=1).to(self.device)
        outp = self.fc2(outp)

        psi_r = outp[:, :self.seq_steps * self.input_size]
        psi_i = outp[:, self.seq_steps * self.input_size:]

        return psi_r, psi_i