import numpy as np


def diff_same_length(y, dx):
    """
    Calculates the Derivative of a function and returns it as numpy array of the same size - the last value is set to zero. 
    """
    
    dy = np.zeros(len(y))
    dy[:-1] = np.diff(y)/dx
    
    return dy


def get_data_paths(toy_data, device, home):
    """
    Returns the right paths for the data folders. Errorchecks, depending on the device.
    """
    
    if (device == 'cpu') and (not toy_data):
        print('You operate on the cpu, sure you dont want to use toy data?(y/n)')
        ans = input()
        if not ans == 'y':
            return None
        
    if (device == 'cuda') and (toy_data):
        print('You operate on cuda, sure you want to use toy data?(y/n)')
        ans = input()
        if not ans == 'y':
            return None    
    
    if toy_data:
        path_initial_data = 'toy_data/' + 'initial_data/'
        path_corrector_data = 'toy_data/' + 'corrector_data/'
    else:
        path_initial_data = home + '/ml_data/data_initial/'
        path_corrector_data = home + '/ml_data/data_corrector/'

    return path_initial_data, path_corrector_data