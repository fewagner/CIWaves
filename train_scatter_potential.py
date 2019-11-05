# ---------------------------------------------
# IMPORTS
# ---------------------------------------------
# libraries
from os.path import expanduser
from types import SimpleNamespace
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

# own files
from lib.LSTMLong import LSTMLong
from lib.utils_scatter import ScatterDataset, Normalize, \
    get_prepared_indices, train, validation, ToTensor

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------

args = SimpleNamespace(epochs=100,
                       batch_size=32,
                       learning_rate=0.001,
                       normalize=True,
                       nr_mean=1.2256541,
                       nr_std=0.17960292,
                       ni_mean=-3.1017997e-05,
                       ni_std=0.027039478,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       which_model='LSTMLong',
                       toy_data=False,
                       amp=1.0,
                       seed=42,
                       log_interval=1
                       )
print('Device: ', args.device)

# ---------------------------------------------
# PATHS
# ---------------------------------------------

home = expanduser("~")
path_model = 'models/scatterpot_' + args.which_model + '.pt'
if not args.toy_data:
    path_data = home + '/ml_data/scatter_potential.hdf5'
    path_here = home + '/MEGAsync/Projektarbeit/Code/'
else:
    path_here = home + '/MEGA/MEGAsync/Projektarbeit/Code/'
# path_data_test = path_here + 'toy_data/initial_data/'
# args.testset_size = 150
path_plots = 'plots/'

# ---------------------------------------------
# CREATE DATASETS
# ---------------------------------------------

transforms = transforms.Compose(
    [Normalize(args),
     ToTensor()])

dataset = ScatterDataset(hdf5_path=path_data,
                         transform=transforms)

print('DATASET CREATED, LENGTH: ', len(dataset))

# ---------------------------------------------
# CREATE DATALOADERS
# ---------------------------------------------

train_indices, validation_indices, test_indices = get_prepared_indices(dataset_size=len(dataset),
                                                                       split_values=[.7, .9],
                                                                       shuffle_dataset=True,
                                                                       random_seed=args.seed)

# Creating PyTorch data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset,
                          batch_size=args.batch_size,
                          sampler=train_sampler)
validation_loader = DataLoader(dataset,
                               batch_size=args.batch_size,
                               sampler=validation_sampler)
test_loader = DataLoader(dataset,
                         batch_size=args.batch_size,
                         sampler=test_sampler)

print('DATALOADERS CREATED:')
print('Samples in Train Set: ', len(train_indices))
print('Samples in Validaiton Set: ', len(validation_indices))
print('Samples in Test Set: ', len(test_indices))
print('\n')

# ---------------------------------------------
# CREATE MODEL
# ---------------------------------------------

model = LSTMLong(input_size=10,
                  hidden_size=100,
                  num_layers=3,
                  seq_steps=25760,
                  device=args.device
                  ).to(args.device)

# ---------------------------------------------
# OPTIMIZER AND LOSS FUNCTION
# ---------------------------------------------

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# ---------------------------------------------
# TRAINING LOOP INCL SAVE, EVAL
# ---------------------------------------------

print('START TRAINING FOR {} EPOCHS.'.format(args.epochs))
print('BATCHSIZE: ', args.batch_size)
print('LEARNING RATE: ', args.learning_rate)
print('MODEL: ', args.which_model)

lowest_loss = 1000000  # for check which model is best
best_epoch = 0


for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, epoch, criterion)
    validation_loss = validation(args, model, validation_loader, criterion)

    # Save best model and plots
    if validation_loss < lowest_loss:
        torch.save(model, path_model)
        print('Model saved.')

    if validation_loss < lowest_loss:
        lowest_loss = validation_loss
        best_epoch = epoch

print("Training done, best epoch was {}.".format(best_epoch))
