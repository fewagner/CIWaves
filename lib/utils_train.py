from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import visdom
import time 
import torch
import csv

from lib.utils_calc_psi import init_val_in, numerov
from lib.utils_evaluation import DeNorm

def get_dataloaders(dataset, batch_size = 8, validation_split = .2, shuffle_dataset = True, random_seed= 42):
    """
    Returns the dataloader objects train_loader and validation_loader. If choosen, shuffels the dataset. Splits the dataset.
    """

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size)) #floor rounds down
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              sampler=train_sampler)
    validation_loader = DataLoader(dataset, 
                                   batch_size=batch_size,
                                   sampler=valid_sampler)
    
    return train_loader, validation_loader


def get_model_loss(model, criterion, train_loader, validation_loader,
                   x, dx, device, nr_mean, nr_std, ni_mean, ni_std, CIWLSTM = False):
    """
    Calculates and returns the loss values für the whole train and validation set
    :param model: the neural network model
    :param criterion: the loss function
    :param train_loader: the dataloader object for the train set
    :param validation_loader: the dataloader object for the validation set
    :return: the loss values for train and validation set
    """
    with torch.no_grad():
        if not CIWLSTM:
            print('checking train loss...')
            train_loss_epoch = 0
            for i, batch in enumerate(train_loader):

                if i%1000 == 0:
                    print('train batch {}/{}'.format(i,len(train_loader)))

                k_n_r = batch['k_n_r'].to(device)
                n_i = batch['n_i'].to(device)
                prediction = model(k_n_r)

                if CIWLSTM:
                    loss = CIWLoss(k_n_r, prediction, criterion, x, dx, device, nr_mean, nr_std, ni_mean, ni_std)
                else:
                    loss = criterion(prediction, n_i)

                train_loss_epoch += loss.item() / len(train_loader)
        else:
            train_loss_epoch = 0

        print('checking validation loss...')
        validation_loss_epoch = 0
        for i, batch in enumerate(validation_loader):

            if i%1000 == 0:
                print('validation batch {}/{}'.format(i,len(validation_loader)))        

            k_n_r = batch['k_n_r'].to(device)
            n_i = batch['n_i'].to(device)
            prediction = model(k_n_r)

            if CIWLSTM:
                loss = CIWLoss(k_n_r, prediction, criterion, x, dx, device, nr_mean, nr_std, ni_mean, ni_std)
            else:
                loss = criterion(prediction, n_i)        

            validation_loss_epoch += loss.item() / len(validation_loader)

    return train_loss_epoch, validation_loss_epoch


def visdom_init(vis, model, criterion, train_loader, validation_loader, x, dx, device, 
                nr_mean, nr_std, ni_mean, ni_std, first_point=True, CIWLSTM = False):
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
        train_loss_epoch, validation_loss_epoch = get_model_loss(model, criterion, train_loader, validation_loader,
                                                                 x, dx, device, nr_mean, nr_std, ni_mean, ni_std, 
                                                                 CIWLSTM = CIWLSTM)

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


def visdom_point(vis, visdom_loss_plot, model, criterion, train_loader, validation_loader, 
                 current_epoch, x, dx, device, nr_mean, nr_std, ni_mean, ni_std, CIWLSTM = False):
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
    train_loss_epoch, validation_loss_epoch = get_model_loss(model, criterion, train_loader, validation_loader,
                                                             x, dx, device, nr_mean, nr_std, ni_mean, ni_std, 
                                                             CIWLSTM = CIWLSTM)

    # füge einen punkt in visdom hinzu
    vis.line(
        X=np.column_stack((current_epoch + 1, current_epoch + 1)),
        Y=np.column_stack((train_loss_epoch,
                           validation_loss_epoch)),
        win=visdom_loss_plot,
        update='append'
    )

    return validation_loss_epoch

def train_net(device, nmbr_epochs, model, criterion, optimizer,
              train_loader, validation_loader, grid, stepsize, nr_mean, nr_std, ni_mean, ni_std,  
              early_stopping=30, path=None, CIWLSTM = False):
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
    if CIWLSTM: 
        first_point = False
    else:
        first_point = True
        
    visdom_loss_plot, info["lowest loss"] = visdom_init(vis, model, criterion, 
                                                        train_loader, validation_loader, 
                                                        grid, stepsize, device, nr_mean, nr_std, ni_mean, ni_std, 
                                                        first_point=first_point, CIWLSTM = CIWLSTM)

    time_start_training = time.time()  # initialize time measurement
    running_loss = 0.0  # loss to print after every batch

    # now training loop
    for current_epoch in range(nmbr_epochs):

        # ------------------------------------
        # TRAINING
        # ------------------------------------

        model = model.train()  # pytorch training mode

        for i, batch in enumerate(train_loader):

            k_n_r = batch['k_n_r'].to(device)
            n_i = batch['n_i'].to(device)

            prediction = model(k_n_r)
            
            if CIWLSTM:
                
                loss = CIWLoss(k_n_r, prediction, criterion, grid, stepsize, device, nr_mean, nr_std, ni_mean, ni_std)

            else:
                
                loss = criterion(prediction, n_i)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()# * k_n_r.size(0)
            if i % 10 == 0:  # print every 10 batches
                print('[%d, %5d] loss: %.8f runtime: %f' %
                      (current_epoch + 1, i + 1, running_loss / 10,
                       time.time() - time_start_training))
                running_loss = 0.0

        # ------------------------------------
        # EVALUATION
        # ------------------------------------

        model = model.eval()  # pytorch evaluation mode

        validation_loss_epoch = visdom_point(vis, visdom_loss_plot, model,
                                             criterion, train_loader, validation_loader, current_epoch,
                                             grid, stepsize, device, nr_mean, nr_std, ni_mean, ni_std, 
                                             CIWLSTM = CIWLSTM)
        
        print('Epoch {} has validation loss {}'.format(current_epoch + 1, validation_loss_epoch))
        
        w = csv.writer(open("info.csv", "w"))
        for key, val in info.items():
            w.writerow([key, val])

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


def calc_psi_Int(x, dx, n, Amp, k):
    """
    Uses the numerov algorithm and the initial values from a wave in empty space to calculate psi and its intensity for wave propagation through a given refraction index. Plots the wave and returns the Intensity as numpy array.
    """
    
    psi_r0, psi_r1, psi_i0, psi_i1 = init_val_in(x,dx,k, Amp)
    
    psi = numerov(x, dx, n, psi_r0, psi_r1, psi_i0, psi_i1, k)
    
    Int = psi.real**2 + psi.imag**2
    
    return psi.real, psi.imag, Int


def CIWLoss(k_n_r, prediction, criterion, grid, stepsize, device, nr_mean, nr_std, ni_mean, ni_std):

    label = torch.zeros(0).to(device)

    for knr,ni in zip(k_n_r,prediction[0]):
        sample_label = torch.zeros(0).to(device)
        sample_label = torch.cat((sample_label,ni),dim=0)

        funcout = calc_psi_Int(grid, stepsize,
                               n = DeNorm(knr.detach().cpu().numpy()[1:],nr_mean,nr_std) + \
                               1j*DeNorm(ni.detach().cpu().numpy(),ni_mean,ni_std), 
                               Amp = 1, 
                               k = DeNorm(knr[0].detach().cpu().numpy(),nr_mean,nr_std))
        sample_label = torch.cat((sample_label,torch.from_numpy(funcout[0]).float().to(device)),dim=0)
        sample_label = torch.cat((sample_label,torch.from_numpy(funcout[1]).float().to(device)),dim=0)
        sample_label = torch.cat((sample_label,torch.ones(funcout[2].shape).to(device)),dim=0)

        label = torch.cat((label,sample_label.view(1,-1)), dim=0)

    prediction_tensor = torch.cat((prediction[0],prediction[1],prediction[2],prediction[3]), dim=1)
        
    loss = criterion(prediction_tensor, label)

    return loss


