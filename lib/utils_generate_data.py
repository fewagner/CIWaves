import numpy as np
import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
from lib.utils_generate_W import generate_W, generate_W_diverse
from lib.utils_general import diff_same_length
from lib.utils_generate_n import n_compl
from lib.utils_calc_psi import plot_psi
from lib.utils_evaluation import get_Int_pred



def generate_init_data(size, path, x, dx):
    """
    Saves size-many numpy arrays of nr and ni in the folder path. In this dataset, the k values are uniformly randomly distributed between 1 and 10. Attention: For size = 5000 the data has about 800 MB.
    """
    
    Potentials = {}
    Potentials['idx'] = []
    Potentials['k'] = []

    start = time.time()
    last = time.time()

    for i in range(size):

        if i%100 == 0: 
            so_far = time.time() - start
            since_last = time.time() - last
            last = time.time()
            print('Generating Pot Nmbr ',i, ', Runtime since last: ', since_last, ', Runtime so far: ', so_far)

        k = np.random.randint(1,11, size=1)[0]

        W = generate_W(x)
        dW = diff_same_length(W, dx)

        n = n_compl(W,dW,x,k)

        np.save(path + 'n_real_' + str(i), n.real)
        np.save(path + 'n_imag_' + str(i), n.imag)
        Potentials['idx'].append(i)
        Potentials['k'].append(k)

    df = pd.DataFrame(Potentials, columns= ['idx', 'k'])

    export_csv = df.to_csv (r''+path+'k_values.csv', index = None, header=True)

    df.head()
    

def generate_init_data_diverse(size, path, x, dx):
    """
    Saves size-many numpy arrays of nr and ni in the folder path. In this dataset, the k values are uniformly randomly distributed between 1 and 10. Attention: For size = 5000 the data has about 800 MB.
    """
    
    Potentials = {}
    Potentials['idx'] = []
    Potentials['k'] = []

    start = time.time()
    last = time.time()
    
    Ws = generate_W_diverse(x)
    dWs = []
    for W in Ws:
        dWs.append(diff_same_length(W, dx))
        
    count = 0

    for i in range(size):

        if i%100 == 0: 
            so_far = time.time() - start
            since_last = time.time() - last
            last = time.time()
            print('Generating Pot Nmbr ',count, ', Runtime since last: ', since_last, ', Runtime so far: ', so_far)

        k = np.random.randint(1,11, size=1)[0]
        
        for W,dW in zip(Ws,dWs):
            n = n_compl(W,dW,x,k)

            np.save(path + 'n_real_' + str(count), n.real)
            np.save(path + 'n_imag_' + str(count), n.imag)
            Potentials['idx'].append(count)
            Potentials['k'].append(k)
            
            count += 1
        
        if np.random.uniform() < 0.1:
            Ws = generate_W_diverse(x)
            dWs = []
            for W in Ws:
                dWs.append(diff_same_length(W, dx))

    df = pd.DataFrame(Potentials, columns= ['idx', 'k'])

    export_csv = df.to_csv (r''+path+'k_values.csv', index = None, header=True)

    df.head()
    
    
def generate_init_data_diverse_wavelendist(size, path, x, dx):
    """
    Saves size-many numpy arrays of nr and ni in the folder path. In this dataset, the k values are uniformly randomly distributed between 1 and 10. Attention: For size = 5000 the data has about 800 MB.
    """
    
    Potentials = {}
    Potentials['idx'] = []
    Potentials['k'] = []

    start = time.time()
    last = time.time()
    
    Ws = generate_W_diverse(x)
    dWs = []
    for W in Ws:
        dWs.append(diff_same_length(W, dx))
        
    count = 0

    for i in range(size):

        if i%100 == 0: 
            so_far = time.time() - start
            since_last = time.time() - last
            last = time.time()
            print('Generating Pot Nmbr ',count, ', Runtime since last: ', since_last, ', Runtime so far: ', so_far)

        k = (2*np.pi/np.random.uniform(2*np.pi/100,2*np.pi,1))[0]
        
        for W,dW in zip(Ws,dWs):
            n = n_compl(W,dW,x,k)

            np.save(path + 'n_real_' + str(count), n.real)
            np.save(path + 'n_imag_' + str(count), n.imag)
            Potentials['idx'].append(count)
            Potentials['k'].append(k)
            
            count += 1
        
        if np.random.uniform() < 0.1:
            Ws = generate_W_diverse(x)
            dWs = []
            for W in Ws:
                dWs.append(diff_same_length(W, dx))

    df = pd.DataFrame(Potentials, columns= ['idx', 'k'])

    export_csv = df.to_csv (r''+path+'k_values.csv', index = None, header=True)

    df.head()    
    
def generate_init_data_diverse_onlyk1(size, path, x, dx):
    """
    Saves size-many numpy arrays of nr and ni in the folder path. In this dataset, the k values are uniformly randomly distributed between 1 and 10. Attention: For size = 5000 the data has about 800 MB.
    """
    
    Potentials = {}
    Potentials['idx'] = []
    Potentials['k'] = []

    start = time.time()
    last = time.time()
    
    Ws = generate_W_diverse(x)
    dWs = []
    for W in Ws:
        dWs.append(diff_same_length(W, dx))
        
    count = 0

    for i in range(size):

        if i%100 == 0: 
            so_far = time.time() - start
            since_last = time.time() - last
            last = time.time()
            print('Generating Pot Nmbr ',count, ', Runtime since last: ', since_last, ', Runtime so far: ', so_far)

        k = 1
        
        for W,dW in zip(Ws,dWs):
            n = n_compl(W,dW,x,k)

            np.save(path + 'n_real_' + str(count), n.real)
            np.save(path + 'n_imag_' + str(count), n.imag)
            Potentials['idx'].append(count)
            Potentials['k'].append(k)
            
            count += 1
        
        #if np.random.uniform() < 0.1:
        Ws = generate_W_diverse(x)
        dWs = []
        for W in Ws:
            dWs.append(diff_same_length(W, dx))

    df = pd.DataFrame(Potentials, columns= ['idx', 'k'])

    export_csv = df.to_csv (r''+path+'k_values.csv', index = None, header=True)

    df.head()
    

def plot_sample(size_dataset, path, x, dx):    
    """
    Load one sample from the dataset in the ordner path which has size size_dataset. Plot the Wave through the potential.
    """
    
    df = pd.read_csv(path + '/k_values.csv')

    idx = np.random.randint(size_dataset, size=1)[0]
    print(idx)

    k = df['k'][idx]

    nr = np.load(path + 'n_real_' + str(idx) + '.npy')
    ni = np.load(path + 'n_imag_' + str(idx) +'.npy')
    n = nr + 1j*ni
    plot_psi(x, dx, n, 1, k)
    
     
def generate_corr_data(model, train_loader, validation_loader, path_corrector_data, 
                       x, dx, nr_mean = 0, nr_std = 1, ni_mean = 0, ni_std = 1):   
    """
    Generate Dataset for the corrector unit. Save data in path_corrector_data.
    """

    count = 0
    keys = ['k_n_r', 'n_i', 'n_i_pred', 'n_i_difference', 'I_pred']
    
    for m,loader in enumerate([train_loader, validation_loader]):
        for i, data in enumerate(loader):
            k_nr = data['k_n_r']
            ni = data['n_i']
            ni_pred = model(k_nr)
            for j in range(len(k_nr)):
                torch.save(k_nr[j], path_corrector_data + keys[0] + '_' + str(count) + '.pt')
                torch.save(ni[j], path_corrector_data + keys[1] + '_' + str(count) + '.pt')
                torch.save(ni_pred[j], path_corrector_data + keys[2] + '_' + str(count) + '.pt')  
                torch.save(ni[j] - ni_pred[j], path_corrector_data + keys[3] + '_' + str(count) + '.pt')
                
                I_pred = get_Int_pred(x, dx, data, ni_pred, j, 1, nr_mean, nr_std, ni_mean, ni_std)
                I_pred = torch.from_numpy(I_pred)
                torch.save(I_pred, 
                           path_corrector_data + keys[4] + '_' + str(count) + '.pt')
                #ni_diff = ni[j] - ni_pred[j]
                count +=1
            print('Saved Loader {} Batch {}'.format(m,i))
        
    return keys
        
        