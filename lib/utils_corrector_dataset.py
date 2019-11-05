import torch
from torch.utils.data import Dataset
import numpy as np


class CorrectorDataset(Dataset):
    """
    Reads in the data for the corrector unit and forms it into a Dataset-Class for PyTorch.
    """
    
    def __init__(self, root_dir, size_dataset, 
                 keys = ['k_n_r', 'n_i', 'n_i_pred', 'n_i_difference', 'I_pred'], transform=None):
        self.root_dir = root_dir
        self.size_dataset = size_dataset
        self.transform = transform
        self.keys = keys
        
    def __len__(self):
        return self.size_dataset
    
    def __getitem__(self, idx):

        sample = {}
        
        for key in self.keys:
            data = torch.load(self.root_dir + key + '_' + 
                            str(idx) + '.pt')
            sample[key] = data
                        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
class NormalizeCorr(object):
    """"
    Data ist then normalized to given mean and std.
    """

    def __init__(self, means, stds):
        # the means and stds are dictionaries with the same keys as every sample has
        self.means = means
        self.stds = stds
    
    def __call__(self, sample):
        
        new_sample = {}
        
        for key in sample.keys():
            new_sample[key] = (sample[key].detach().cpu().float() - self.means[key].float())/self.stds[key].float()
               
        return new_sample    
    
    
def calc_mean_std_corr(path_data, size_dataset, keys):
    """
    load the size numpy arrays from directory path and calculate mean and std of data, return dictionaries with the values
    """
    
    means = {}
    stds = {}

    for key in keys:
        means[key] = np.zeros(1)
        stds[key] = np.zeros(1)
        for i in range(size_dataset):
            data = torch.load(path_data + key + '_' + str(i) + '.pt')
            means[key] += np.mean(data.detach().cpu().numpy()) 
            stds[key] += np.std(data.detach().cpu().numpy())
        means[key] = torch.from_numpy(means[key]/size_dataset)
        stds[key] = torch.from_numpy(stds[key]/size_dataset)
            
    return means, stds
    
    
class ToDevice(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        
        new_sample = {}
        
        for key in sample.keys():
            new_sample[key] = sample[key].float().to(self.device)
        
        return new_sample
    
    