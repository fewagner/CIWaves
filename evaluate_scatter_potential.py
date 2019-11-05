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
import matplotlib.pyplot as plt

# own files
from lib.LSTMLong import LSTMLong
from lib.utils_scatter import ScatterDataset, Normalize, \
    get_prepared_indices, train, validation, ToTensor, get_coefficients
from lib.utils_test_model import DN

# ---------------------------------------------
# PARAMETERS
# ---------------------------------------------

args = SimpleNamespace(batch_size=10,
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
                       log_interval=1,
                       N=257600,
                       d=0.0006250024262516547,
                       eval_hermitean=True,
                       eval_prediction=True,
                       eval_label=True
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
test_sampler = SubsetRandomSampler(test_indices)

test_loader = DataLoader(dataset,
                         batch_size=args.batch_size,
                         sampler=test_sampler)

print('DATALOADER CREATED:')
print('Samples in Test Set: ', len(test_indices))
print('\n')

# ---------------------------------------------
# LOAD MODEL
# ---------------------------------------------

model = torch.load(path_model,
                   map_location=args.device)
model.device = args.device
model.to(args.device)
print('Model on CUDA: ', next(model.parameters()).is_cuda)
model.eval()

# ---------------------------------------------
# EVALUATION LOOP, CALC SOME REFLECTION INDICES
# ---------------------------------------------

print('START EVALUATION.')
print('BATCHSIZE: ', args.batch_size)
print('MODEL: ', args.which_model)

criterion = torch.nn.MSELoss()

if args.eval_prediction:
    file_prediction = open('plots/scattercoefficients_prediction.txt','w+')
    file_prediction.write('idx\tt_l\tr_l\tt_r\tr_r\n')
if args.eval_hermitean:
    file_hermitean = open('plots/scattercoefficients_hermitean.txt','w+')
    file_hermitean.write('idx\tt_l\tr_l\tt_r\tr_r\n')
if args.eval_label:
    file_label = open('plots/scattercoefficients_label.txt','w+')
    file_label.write('idx\tt_l\tr_l\tt_r\tr_r\n')

for batch_idx, data in enumerate(test_loader):
    nr = data['n_r'].to(args.device)
    k = data['k'].to(args.device)
    input = torch.cat((k, nr), dim=1)
    label = data['n_i'].to(args.device)
    output = model(input)

    # calculate mse loss
    loss = criterion(output, label)
    print('mse error: ', loss.item())

    for i, ni in enumerate(output):

        print('Potential ', batch_idx * args.batch_size + i)

        if args.eval_prediction:
            t_l, r_l, t_r, r_r = get_coefficients(N=args.N,
                                                  k=k[i].detach().cpu().numpy()[0],
                                                  n=DN(nr[i].detach().cpu().numpy(), 'r', args)
                                                    + 1j * DN(ni.detach().cpu().numpy(), 'i', args),
                                                  d=args.d)

            file_prediction.write('{}\t{}\t{}\t{}\t{}\n'.format(batch_idx * args.batch_size + i, t_l, r_l, t_r, r_r))

        if args.eval_hermitean:
            t_l, r_l, t_r, r_r = get_coefficients(N=args.N,
                                                  k=k[i].detach().cpu().numpy()[0],
                                                  n=DN(nr[i].detach().cpu().numpy(), 'r', args),
                                                  d=args.d)

            file_hermitean.write('{}\t{}\t{}\t{}\t{}\n'.format(batch_idx * args.batch_size + i, t_l, r_l, t_r, r_r))

        if args.eval_label:
            t_l, r_l, t_r, r_r = get_coefficients(N=args.N,
                                                  k=k[i].detach().cpu().numpy()[0],
                                                  n=DN(nr[i].detach().cpu().numpy(), 'r', args)
                                                    + 1j * DN(label[i].detach().cpu().numpy(), 'i', args),
                                                  d=args.d)

            file_label.write('{}\t{}\t{}\t{}\t{}\n'.format(batch_idx * args.batch_size + i, t_l, r_l, t_r, r_r))
    #break

if args.eval_prediction:
    file_prediction.close()
if args.eval_hermitean:
    file_hermitean.close()
if args.eval_label:
    file_label.close()

print("Evaluation done.")
