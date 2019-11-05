# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

# official libraries
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn as nn
from os.path import expanduser
from types import SimpleNamespace

# own files
from lib.utils_ODEMultistep import validation, train, ODEMultistep
from lib.utils_ODEOpt import CIDataset
from lib.utils_generate_W import get_grid
from lib.utils_initial_dataset import ToTensor, calc_mean_std
from lib.utils_train import get_dataloaders
from lib.utils_test_model import Norm

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------

args = SimpleNamespace(epochs=500,
                       lr=0.0001,
                       batch_size=5,
                       normalize=False,
                       dataset_size=150000,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       toy_data=False,
                       load_model=True,
                       log_interval=1,
                       half_width=5,
                       nmbr_points=10000,
                       gauss_data=True,
                       validation_split=0.5,
                       steps=4,
                       which_model='ODEMultistep',
                       seed=42,
                       taylor=False
                       )

args.grid, args.stepsize = get_grid(args.half_width, args.nmbr_points)
print('DEVICE: ', args.device)
print('LEARNING RATE: ', args.lr)
print('EPOCHS: ', args.epochs)
print('BATCH SIZE: ', args.batch_size)
print('NORMALIZE: ', args.normalize)
print('TOY DATA: ', args.toy_data)
print('LOAD MDOEL: ', args.load_model)
print('GAUSS DATA: ', args.gauss_data)

# -------------------------------------------------
# PATHS
# -------------------------------------------------

home = expanduser("~")
path_model = 'models/' + args.which_model + '.pt'
if not args.toy_data:
    if args.gauss_data:
        path_data = 'toy_data/data_gauss/'
    else:
        path_data = home + '/ml_data/data_initial/'
    path_here = home + '/MEGAsync/Projektarbeit/Code/'
else:
    path_here = home + '/MEGA/MEGAsync/Projektarbeit/Code/'
    if args.gauss_data:
        path_data = 'toy_data/data_gauss/'
    else:
        path_data = path_here + 'toy_data/initial_data/'
    args.testset_size = 150
path_plots = 'plots/'

# -------------------------------------------------
# CREATE DATASET
# -------------------------------------------------

if args.normalize:
    args.nr_mean, args.nr_std, \
    args.ni_mean, args.ni_std = calc_mean_std(args.testset_size,
                                              path_data)
    transform = transforms.Compose([Norm(args),
                                    ToTensor()
                                    ])
else:
    transform = transforms.Compose([ToTensor()
                                    ])

dataset = CIDataset(csv_file=path_data + 'k_values.csv',
                    root_dir=path_data,
                    transform=transform
                    )
print('DATASET LOADED.')
print('LENGTH DATASET: ', len(dataset))

# -------------------------------------------------
# CREATE DATALOADERS
# -------------------------------------------------

train_loader, validation_loader = get_dataloaders(dataset,
                                                  batch_size=args.batch_size,
                                                  validation_split=args.validation_split,
                                                  shuffle_dataset=True,
                                                  random_seed=args.seed)

# -------------------------------------------------
# CREATE OR LOAD MODEL
# -------------------------------------------------

if args.load_model:
    print('LOADING MODEL.')
    model = torch.load(path_model, map_location=args.device)
    model.to(args.device)
    model.device = args.device
else:
    print('CREATING MODEL.')
    model = ODEMultistep(args=args,
                         taylor=args.taylor)

print('MODEL: ', model)
print('MDOEL ON CUDA: ', next(model.parameters()).is_cuda)

# -------------------------------------------------
# TRAINING
# -------------------------------------------------

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lowest_loss = 1000000  # for check which model is best
best_epoch = 0

for epoch in range(1, args.epochs + 1):
    train(args, model, criterion, train_loader, optimizer, epoch)
    validation_loss = validation(args, model, criterion, validation_loader)

    # Save best model
    if validation_loss < lowest_loss:
        torch.save(model, path_model)
        print('Model saved.')

        print('PLOTTING PREDICTION.')

        data = next(iter(validation_loader))
        nr = data['k_n_r'][0, 1:].to(args.device).view(1, -1)
        ni = data['n_i'][0].to(args.device).view(1, -1)
        k = data['k_n_r'][0, 0].to(args.device).view(1, -1)
        iv = ni[:, :args.steps]

        ni_pred = model(nr=nr,
                        iv=iv,
                        k=k,
                        stop=len(args.grid)
                        )

        plt.close('all')
        plt.plot(args.grid, nr.view(-1).detach().cpu().numpy(), label='nr')
        plt.plot(args.grid, ni.view(-1).detach().cpu().numpy(), label='ni')
        plt.plot(args.grid, ni_pred.view(-1).detach().cpu().numpy(), label='ni_pred')
        plt.savefig(path_plots + 'gauss.pdf')

        lowest_loss = validation_loss
        best_epoch = epoch

print("Training done, best epoch was {}.".format(best_epoch))
