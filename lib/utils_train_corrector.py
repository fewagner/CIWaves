from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import visdom
import time 
import torch

def get_FNN_input(batch):
    I_pred = batch['I_pred']
    k_n_r = batch['k_n_r']
    n_i_pred = batch['n_i_pred']
    data = torch.cat( (I_pred, k_n_r, n_i_pred) , dim=1)
    
    return data


def get_model_loss(model, criterion, train_loader, validation_loader, which_model):
    """
    Calculates and returns the loss values für the whole train and validation set
    :param model: the neural network model
    :param criterion: the loss function
    :param train_loader: the dataloader object for the train set
    :param validation_loader: the dataloader object for the validation set
    :return: the loss values for train and validation set
    """

    losses = [0,0]
    data_loaders = [train_loader, validation_loader]
    
    if which_model == 'FNN':
        for j,loader in enumerate(data_loaders):        
            for i, batch in enumerate(loader):
                data = get_FNN_input(batch)
                ni_diff = batch['n_i_difference']
                prediction = model(data)
                losses[j] += criterion(prediction, ni_diff).item() / len(loader) * data.size(0)
    elif which_model == 'UNet':
                print('todo') # todo! Inputs.... wie?
    else:
        raise Exception('Model not implemented!')
        
    return losses[0], losses[1]


def visdom_init(vis, model, criterion, train_loader, validation_loader, which_model, first_point=True):
    """
    Initialize Visdom Window and print first two, initial loss points
    :param vis: link to initialized visdom instance
    :param model: the pytorch model
    :param criterion: the initialized loss function
    :param train_loader: the train loader
    :param validation_loader: the validation loader
    :param first_point: if true, the visdom monitor is initialized with an initial loss point with the untrained model
    :return: returns the visdom window for the loss lines and the validation loss
    """

    if first_point:
        train_loss_epoch, validation_loss_epoch = get_model_loss(model, criterion, train_loader, validation_loader, which_model)

    else:
        train_loss_epoch, validation_loss_epoch = 0, 0

    visdom_loss_plot = vis.line(
        X=np.column_stack((0, 0)),
        Y=np.column_stack((train_loss_epoch, validation_loss_epoch)),
        opts={
            'title': 'Loss Event-Dataset',
            'legend': ['train', 'validation']}
    )

    return visdom_loss_plot, validation_loss_epoch


def visdom_point(vis, visdom_loss_plot, model, criterion, train_loader, validation_loader, current_epoch, which_model):
    """
    Plots two loss points into an initialized visdom window
    :param vis: link to initialized visdom instance
    :param visdom_loss_plot: visdom window, initialized by visdom_init
    :param model: the pytorch model
    :param criterion: the initialized loss function
    :param train_loader: the train loader
    :param validation_loader: the validation loader
    :param ep: number of current epoch, needed for the loss plot
    :return: the validation loss e.g. for early stopping
    """

    # print learning curve with visdom
    train_loss_epoch, validation_loss_epoch = get_model_loss(model, criterion, train_loader, validation_loader, which_model)

    # füge einen punkt in visdom hinzu
    vis.line(
        X=np.column_stack((current_epoch + 1, current_epoch + 1)),
        Y=np.column_stack((train_loss_epoch,
                           validation_loss_epoch)),
        win=visdom_loss_plot,
        update='append'
    )

    return validation_loss_epoch


def train_corrector(nmbr_epochs, model, criterion, optimizer,
              train_loader, validation_loader, which_model, 
              early_stopping=30, path=None):
    """
    Trains a PyTorch model for some number of epochs.
    Needs started visdom!
    Early stopping is implemented and the best model is always saved
    :param nmbr_epochs: the number of epochs the model shall be trained
    :param model: the pytorch model to train
    :param criterion: the initialized loss function
    :param optimizer: the initialized optimizer
    :param early_stopping: number after how many epochs without improvement the training should stop
        (if 0 - no early stopping)
    :param path: the path where the model is saved
    :return: trained model
    """

    # Initialize Info Dict
    info = {"lowest loss": None,
            "last epoch": 0
            }

    # Initialize Visdom Loss Monitor
    vis = visdom.Visdom()
    visdom_loss_plot, info["lowest loss"] = visdom_init(vis, model, criterion,
                                                        train_loader, validation_loader, 
                                                        which_model, first_point=True)

    time_start_training = time.time()  # initialize time measurement
    running_loss = 0.0  # loss to print after every batch

    # now training loop
    for current_epoch in range(nmbr_epochs):

        # ------------------------------------
        # TRAINING
        # ------------------------------------

        model = model.train()  # pytorch training mode

        if which_model == 'FNN':

            for i, batch in enumerate(train_loader):

                data = get_FNN_input(batch)
                ni_diff = batch['n_i']

                prediction = model(data)
                loss = criterion(prediction, ni_diff)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item() * data.size(0)
                if i % 10 == 0:  # print every 10 batches
                    print('[%d, %5d] loss: %.3f runtime: %f' %
                          (current_epoch + 1, i + 1, running_loss / 10,
                           time.time() - time_start_training))
                    running_loss = 0.0
                    
        elif which_model == 'UNet':
            
            print('todo')
            # todo!
            
        else:
            raise Exception('Model not implemented!')

        # ------------------------------------
        # EVALUATION
        # ------------------------------------

        model = model.eval()  # pytorch evaluation mode

        validation_loss_epoch = visdom_point(vis, visdom_loss_plot, model,
                                             criterion, train_loader, validation_loader, 
                                             current_epoch, which_model)

        # ------------------------------------
        # SAVE MODEL AND EARLY STOPPING
        # ------------------------------------

        if validation_loss_epoch < info["lowest loss"]:
            info["lowest loss"] = validation_loss_epoch
            info["last epoch"] = current_epoch + 1
            if path != None:
                torch.save(model, path)
                print("MODEL SAVED.")
            else:
                print("NO PATH FOR SAVING SET!")

        elif (current_epoch > info["last epoch"] + early_stopping) and (early_stopping > 0):
            print("STOPPED EARLY AFTER EPOCH {}.".format(current_epoch + 1))
            break

    return model, info
