import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class InitialDataset(Dataset):
    """
    Reads in the initial data and forms it into a Dataset-Class for PyTorch.
    """
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.k_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.k_frame)
    
    def __getitem__(self, idx):
        nr = np.load(self.root_dir + 'n_real_' + str(idx) +'.npy')
        ni = np.load(self.root_dir + 'n_imag_' + str(idx) +'.npy')
        k = self.k_frame['k'][idx]
        
        k_nr = np.append(k,nr)
        
        sample = {'k_n_r': k_nr, 
                  'n_i': ni}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    #def __init__(self, device):
    #    self.device = device

    def __call__(self, sample):
        
        new_sample = {}
        
        for key in sample.keys():
            new_sample[key] = torch.from_numpy(sample[key]).float()#.to(self.device)
        
        return new_sample

        #k_nr, ni = sample['k_n_r'], sample['n_i']
        
        #return {'k_n_r': torch.from_numpy(k_nr).float().to(self.device),
        #        'n_i': torch.from_numpy(ni).float().to(self.device)}
    
    
class OneChannel(object):
    """For Convolution, size (N,C,L) is needed. This class is not used anymore."""

    def __init__(self, device):
        self.device = device    
    
    def __call__(self, sample):
        new_samp = {'k_n_r': torch.ones([1,len(sample['k_n_r'])]),
                    'n_i': torch.ones([1,len(sample['n_i'])])}
        new_samp['k_n_r'][0] = sample['k_n_r']
        new_samp['n_i'][0] = sample['n_i']
        new_samp['k_n_r'] = new_samp['k_n_r'].to(self.device)
        new_samp['n_i'] = new_samp['n_i'].to(self.device)
        
        return new_samp
    
    
class Normalize(object):
    """"Data ist then normalized to given mean and std."""

    def __init__(self, means, stds):
        self.nr_mean = means[0]
        self.nr_std = stds[0]
        self.ni_mean = means[1]
        self.ni_std = stds[1]
    
    def __call__(self, sample):
        k_nr, ni = sample['k_n_r'], sample['n_i']
        
        k_nr = (k_nr - self.nr_mean)/self.nr_std
        ni = (ni - self.ni_mean)/self.ni_std
        
        return {'k_n_r': k_nr,
                'n_i': ni}
    
    
def calc_mean_std(size, path):
    """
    load the size numpy arrays from directory path and calculate mean and std of data, return the four values
    """

    nr_mean = 0
    nr_std = 0
    ni_mean = 0
    ni_std = 0

    for i in range(size):
        nr = np.load(path + 'n_real_' + str(i) + '.npy')
        ni = np.load(path + 'n_imag_' + str(i) + '.npy')

        nr_mean += np.mean(nr)
        nr_std += np.std(nr)
        ni_mean += np.mean(ni)
        ni_std += np.std(ni)

    nr_mean = nr_mean/size
    nr_std = nr_std/size
    ni_mean = ni_mean/size
    ni_std = ni_std/size    

    return nr_mean, nr_std, ni_mean, ni_std
