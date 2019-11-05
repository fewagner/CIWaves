# ---------------------------------------------
# IMPORTS
# ---------------------------------------------

from os.path import expanduser
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time

from lib.utils_general import get_data_paths
from lib.utils_initial_dataset import InitialDataset, ToTensor
from lib.utils_train import get_dataloaders
from lib.utils_ODEOpt import PyNumerov
from lib.utils_calc_psi import init_val_in
from lib.utils_generate_W import get_grid
from lib.utils_calc_psi import plot_psi
from lib.ODESolver import ODESolver

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------

args = SimpleNamespace(lr=0.0001,
                       batch_size=8,
                       epochs=5,
                       seed=42,
                       toy_data=False,
                       load_model=True,
                       train_model=True,
                       which_model='ODESolver',
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       steps=4,
                       start_epoch=2, # default to 1
                       start_loss= 0.0038936228262881436, # default to some high number
                       trainset_size = 4000,
                       validationset_size = 1000)

print('DEVICE: ', args.device)

if args.toy_data:
    size_dataset = 8  # if run on GPU machine, set to 5000 (toydata 8)
else:
    size_dataset = 10000
args.grid, args.stepsize = get_grid(half_width=5,
                                    nmbr_points=10000)
print('PARAMETERS: ', args)

# ---------------------------------------------
# PATHS
# ---------------------------------------------

path_model = 'models/' + args.which_model + '.pt'
home = expanduser("~")
path_plots = 'plots/'
# path_data, _ = get_data_paths(args.toy_data,
#                               args.device,
#                               home)
if args.toy_data == True:
    path_data = 'toy_data/initial_data/'
else:
    path_data = home + '/ml_data/data_initial_diverse/'
    #path_test_data = home + '/ml_data/data_test/'

# ---------------------------------------------
# IMPORT DATA
# ---------------------------------------------

print('\nIMPORT DATA:')

transform = transforms.Compose([ToTensor()  # no normalize so far
                                ])

dataset = InitialDataset(csv_file=path_data + 'k_values.csv',
                         root_dir=path_data,
                         transform=transform)
print('Dataset Length: ', len(dataset))
print('Stopping Training after {} samples.'.format(args.trainset_size))
print('Stopping Validation after {} samples.'.format(args.validationset_size))

#if not args.toy_data:
#    testset = InitialDataset(csv_file=path_test_data + 'k_values.csv',
#                             root_dir=path_test_data,
#                             transform=transform)
#    print('Testset Length: ', len(testset))

# ---------------------------------------------
# CREATE DATALOADERS
# ---------------------------------------------

train_loader, validation_loader = get_dataloaders(dataset,
                                                  batch_size=args.batch_size,
                                                  validation_split=0.2,
                                                  shuffle_dataset=True,
                                                  random_seed=42)

#if not args.toy_data:
#    test_loader, _ = get_dataloaders(testset,
#                                     batch_size=args.batch_size,
#                                     validation_split=0,
#                                     shuffle_dataset=False,
#                                     random_seed=42)

# ---------------------------------------------
# LOAD OR CREATE MODEL
# ---------------------------------------------

if args.load_model:
    model = torch.load(path_model, map_location=args.device)
elif args.which_model == 'ODESolver':
    model = ODESolver(args.steps,
                      n_hidden=500)
    print(model)
    model.to(args.device)
    print('Model on CUDA: ', next(model.parameters()).is_cuda)
else:
    print('No model created or loaded.')

# ---------------------------------------------
# TRAINING
# ---------------------------------------------

print('TRAINING FOR {} EPOCHS.'.format(args.epochs))

if args.train_model:
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    info = {
        "lowest loss": args.start_loss,
        "last epoch": args.start_epoch-1,
    }

    # init runtime measurement
    time_start_training = time.time()

    # now training loop
    for current_epoch in range(args.start_epoch, args.epochs + args.start_epoch):

        # ------------------------------------
        # TRAINING
        # ------------------------------------

        model = model.train()

        running_loss = 0

        for i, batch in enumerate(train_loader):

            if i > args.trainset_size/args.batch_size:
                    break

            k = batch['k_n_r'][:, 0].to(args.device)
            nr = batch['k_n_r'][:, 1:].to(args.device)
            ni = batch['n_i'].to(args.device)
            ni_pred = torch.zeros(ni.size()).to(args.device)

            for j in range(len(args.grid) - args.steps - 1):
                input = torch.zeros(len(ni), 2 * args.steps + 3).to(args.device)

                input[:, 0] = k
                input[:, 1] = args.stepsize
                input[:, 2:2 + args.steps + 1] = nr[:, j:j + args.steps + 1]
                input[:, 3 + args.steps:] = ni[:, j:j + args.steps]

                # in the very spirit of ODE Solver Methods
                ni_pred[:, j + args.steps + 1] = ni[:, j + args.steps] + args.stepsize * model(input).view(-1)

                # if j % 1000 == 0:
                #     print(j)

            #loss = criterion(ni[:, j + args.steps + 1], ni_pred[:, j + args.steps + 1])
            loss = criterion(ni, ni_pred)
            optimizer.zero_grad()
            loss.backward() # retain_graph=True
            optimizer.step()
            running_loss += loss.item()

            # print statistics
            #if i % 10 == 0:  # print every 10 batches
            print('[%d, %5d] l_model: %f time: %.1f' %
                  (current_epoch, i + 1, running_loss,
                   time.time() - time_start_training))
            running_loss = 0

        # ------------------------------------
        # EVALUATION
        # ------------------------------------

        model = model.eval()

        validation_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(validation_loader):

                if i > args.validationset_size/args.batch_size:
                    break

                k = batch['k_n_r'][:, 0].to(args.device)
                nr = batch['k_n_r'][:, 1:].to(args.device)
                ni = batch['n_i'].to(args.device)
                ni_pred = torch.zeros(ni.size()).to(args.device)

                for j in range(len(args.grid) - args.steps - 1):
                    input = torch.zeros(len(ni), 2 * args.steps + 3).to(args.device)

                    input[:, 0] = k
                    input[:, 1] = args.stepsize
                    input[:, 2:2 + args.steps + 1] = nr[:, j:j + args.steps + 1]
                    input[:, 3 + args.steps:] = ni[:, j:j + args.steps]

                    ni_pred[:, j + args.steps + 1] = ni[:, j + args.steps] + args.stepsize * model(input).view(-1)

                    # if j % 1000 == 0:
                    #     print(j)

                validation_loss += criterion(ni, ni_pred).item()

        validation_loss /= len(validation_loader)
        print('Epoch {} has validation loss {}'.format(current_epoch,
                                                       validation_loss))

        if validation_loss < info["lowest loss"]:
            info["lowest loss"] = validation_loss
            info["last epoch"] = current_epoch
            torch.save(model, path_model)
            print("MODEL SAVED.")

    print('Training done.')

print('\nDONE.')
