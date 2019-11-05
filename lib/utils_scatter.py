import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


# ------------------------------------------------------
# DATASET STUFF
# ------------------------------------------------------

class ScatterDataset(Dataset):

    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.f = h5py.File(hdf5_path, 'r')
        self.group_keys = list(self.f.keys())
        self.len = len(list(self.f[self.group_keys[0]]))  # key 0 are the displacements
        self.nr_index = self.group_keys.index('n_r')
        self.ni_index = self.group_keys.index('n_i')
        self.psir_index = self.group_keys.index('psi_r')
        self.psii_index = self.group_keys.index('psi_i')
        self.k_index = self.group_keys.index('k')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        n_r = np.array(self.f[self.group_keys[self.nr_index]][idx])
        n_i = np.array(self.f[self.group_keys[self.ni_index]][idx])  # needs a number not 1D arrays
        psi_r = np.array(self.f[self.group_keys[self.psir_index]][idx])
        psi_i = np.array(self.f[self.group_keys[self.psii_index]][idx])  # needs a number not 1D arrays
        k = np.array(self.f[self.group_keys[self.k_index]][idx])

        sample = {'n_r': n_r,
                  'n_i': n_i,
                  'psi_r': psi_r,
                  'psi_i': psi_i,
                  'k': k
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize(object):
    """
    Normalize Tensor to given mean and std
    """

    def __init__(self, args):
        self.nr_mean = args.nr_mean
        self.nr_std = args.nr_std
        self.ni_mean = args.ni_mean
        self.ni_std = args.ni_std

    def __call__(self, sample):
        new_sample = sample
        new_sample['n_r'] = (sample['n_r'] - self.nr_mean) / self.nr_std
        new_sample['n_i'] = (sample['n_i'] - self.ni_mean) / self.ni_std
        return new_sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    # def __init__(self, device):
    #    self.device = device

    def __call__(self, sample):
        new_sample = {}

        for key in sample.keys():
            new_sample[key] = torch.from_numpy(sample[key])  # .to(self.device)

        return new_sample


def get_prepared_indices(dataset_size, split_values, shuffle_dataset=True, random_seed=42):
    """
    Chooses the indices for the Split datasets.
    :param dataset_size: Size of the whole dataset, is a number
    :param split_values: a list with the values at which the indices should be split.
        f.i. [0.5,0.7] creates 50/20/30 split
    :param shuffle_dataset: When true, the indices are dataset is shuffled befor the indices are assigned
    :param random_seed: set of some value to get the same datasets always for comparability
    :return: indices for training, validation and test set
    """

    indices = list(range(dataset_size))
    split = [int(split_values[0] * dataset_size), int(split_values[1] * dataset_size)]  # floor rounds down
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, validation_indices, test_indices = indices[:split[0]], indices[split[0]:split[1]], indices[split[1]:]

    return train_indices, validation_indices, test_indices


# ------------------------------------------------------
# TRAIN FUNCTION
# ------------------------------------------------------

def train(args, model, train_loader, optimizer, epoch, criterion):
    model.train()
    print('\n')
    running_loss = 0
    for batch_idx, data in enumerate(train_loader):
        nr = data['n_r'].to(args.device)
        k = data['k'].to(args.device)
        input = torch.cat((k, nr), dim=1)
        label = data['n_i'].to(args.device)
        output = model(input)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                running_loss / args.log_interval))
            running_loss = 0


# ------------------------------------------------------
# VALIDATION FUNCTION
# ------------------------------------------------------

def validation(args, model, validation_loader, criterion):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            nr = data['n_r'].to(args.device)
            k = data['k'].to(args.device)
            input = torch.cat((k, nr), dim=1)
            label = data['n_i'].to(args.device)
            output = model(input)
            loss = criterion(output, label)
            loss = loss.item()
            validation_loss += loss
    validation_loss /= len(validation_loader)
    print('Validation set (mse) loss: {:.6f}'.format(validation_loss))

    return validation_loss


# ------------------------------------------------------
# TRANSFER MATRIX METHOD
# ------------------------------------------------------

def get_coefficients(N, k, n, d):
    # first calculate the M matrix for the whole potential

    M = np.zeros([N, 2, 2], dtype=complex)
    M[0] = [[np.exp(1j * k * n[0] * d), 0],
            [0, np.exp(-1j * k * n[0] * d)]]

    for i in range(1, N):
        D0 = [[1, 1],
              [n[i - 1], -n[i - 1]]]
        M[i] = np.matmul(D0, M[i - 1])

        D1 = [[1, 1],
              [n[i], -n[i]]]
        D1 = np.linalg.inv(D1)
        M[i] = np.matmul(D1, M[i])

        phases = np.array([[np.exp(1j * k * n[i] * d), 0],
                           [0, np.exp(-1j * k * n[i] * d)]])
        # print('phases: ', phases)
        # print('M[i]: ', M[i])
        M[i] = np.matmul(phases, M[i])

    # now calcualte S matrix for whole potential

    S = np.zeros([2, 2], dtype=complex)
    M_all = M[-1]
    S[0, 0] = - M_all[1, 0] / M_all[0, 0]
    S[0, 1] = M_all[1, 1] - (M_all[0, 1] * M_all[1, 0]) / M_all[0, 0]
    S[1, 0] = 1 / M_all[0, 0]
    S[1, 1] = - M_all[0, 1] / M_all[1, 1]

    t_l = np.abs(S[1, 0]) ** 2
    r_l = np.abs(S[0, 0]) ** 2
    t_r = np.abs(S[0, 1]) ** 2
    r_r = np.abs(S[1, 1]) ** 2

    return t_l, r_l, t_r, r_r
