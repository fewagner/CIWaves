import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import visdom

from lib.utils_stack_model_psisolver import get_psi
from lib.utils_evaluation import DeNorm


def corrector_init_visdom():
    
    vis = visdom.Visdom()
    
    train_model_loss = 1
    validation_model_loss = 1
    train_corrector_loss = 1
    validation_corrector_loss = 1
    
    visdom_model = vis.line(
        X=np.column_stack((0, 0)),
        Y=np.column_stack((train_model_loss,
                           validation_model_loss)),
        opts={
            'title': 'Model Loss',
            'legend': ['ni_train', 'Int_train', 'ni_validation', 'Int_validation']}
    )
    
    visdom_corrector = vis.line(
        X=np.column_stack((0, 0)),
        Y=np.column_stack((train_corrector_loss,
                           validation_corrector_loss)),
        opts={
            'title': 'Corrector Loss',
            'legend': ['train', 'validation']}
    )
    
    return vis, visdom_model, visdom_corrector


def corrector_train(criterion,
                   optimizer,
                   prediction,
                   label,
                   retain_graph=False):
    
    loss = criterion(prediction, label)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    
    return loss.item()


def corrector_visdom_point(vis, 
                           current_epoch,
                           visdom_model, 
                           visdom_corrector, 
                           validation_loader,
                           criterion, 
                           train_model_loss, 
                           train_corrector_loss,
                           grid,
                           stepsize,
                          device,
                          model,
                          corrector,
                            nr_mean,
                            nr_std,
                            ni_mean,
                            ni_std):

    validation_model_loss = 0
    validation_corrector_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(validation_loader):

            if i % 100 == 0:
                print('validation batch {}/{}'.format(i, len(validation_loader)))

            # unravel data
            k_n_r = batch['k_n_r'].to(device)
            n_i = batch['n_i'].to(device)

            # make predictions and calc psi and Int
            ni_pred = model(k_n_r)

            _, _, Int = get_psi(grid,
                             stepsize,
                             DeNorm(k_n_r, nr_mean, nr_std),
                             DeNorm(ni_pred, ni_mean, ni_std),
                             Amp=1)

            Int = Int.to(device)

            corrector_input = torch.cat((k_n_r[:,0].view(-1,1), Int),dim=1)

            ni_diff_pred = model(corrector_input)
            ni_diff = n_i - ni_diff_pred

            # sum up losses
            validation_model_loss += \
                criterion(n_i, ni_pred).item()/len(validation_loader)

            validation_corrector_loss += \
                criterion(ni_diff, ni_diff_pred).item()/len(validation_loader)

            # add visdom point
            vis.line(X=np.column_stack((current_epoch + 1, current_epoch + 1)),
                     Y=np.column_stack((train_model_loss, validation_model_loss)),
                     win=visdom_model,
                     update='append')

            vis.line(X=np.column_stack((current_epoch + 1, current_epoch + 1)),
                     Y=np.column_stack((train_corrector_loss, validation_corrector_loss)),
                     win=visdom_corrector,
                     update='append')

    return validation_model_loss, \
           validation_corrector_loss