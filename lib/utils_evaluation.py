import matplotlib.pyplot as plt
from lib.utils_train_corrector import get_FNN_input
from lib.utils_calc_psi import plot_psi


def DeNorm(x,mean,std):
    """
    Takes a normalized sample, together with mean and std to that it should be scaled back and returns the original sample.
    """
    
    x = x*std + mean
    return x


def plot_prediction(dataloader, model, x, nr_mean = 0, nr_std = 1, ni_mean = 0, ni_std = 1, idx=None, k=None, device='cpu'):
    """
    Plots the predicted ni vs the label ni from a random batch from the dataloader. The index inside the batch can be choosen individually or randomly, as well as the k values can be choosen individually or taken from the dataset. Returns all plotted values.
    """
    
    dataiter = iter(dataloader)
    item = dataiter.next()

    if idx == None:
        idx = np.random.randint(0,len(item), size=1)[0]
    
    k_nr = item['k_n_r'][idx]
    
    if k == None: # k can be choosen individually or taken from the dataset
        k = DeNorm(k_nr[0].cpu().numpy(), nr_mean, nr_std)
    else:
        item['k_n_r'][idx,0] = k
        
    nr = k_nr[1:].cpu().numpy()
    nr = DeNorm(nr, nr_mean, nr_std)
    ni = item['n_i'][idx].cpu().numpy()
    ni = DeNorm(ni, ni_mean, ni_std)

    ni_pred = model(item['k_n_r'].to(device)).detach().cpu().numpy()[0]
    ni_pred = DeNorm(ni_pred, ni_mean, ni_std)

    print(k) #the higher k, the smaller the ni peaks

    plt.plot(x,nr,label='n_r')
    plt.plot(x,ni_pred,label='n_i prediction')
    plt.plot(x,ni,label='n_i label')
    plt.legend(loc='upper left')
    
    return nr, ni, ni_pred, k


def plot_correction(dataloader, model, x, means, stds, idx=None, k=None):
    """
    todo!
    Plots the predicted ni vs the label ni from a random batch from the dataloader. The index inside the batch can be choosen individually or randomly, as well as the k values can be choosen individually or taken from the dataset. Returns all plotted values.
    """
    
    
    dataiter = iter(dataloader)
    item = dataiter.next()

    if idx == None:
        idx = np.random.randint(0,len(item), size=1)[0]
    
    data = get_FNN_input(item)[idx]
       
    if k == None: # k can be choosen individually or taken from the dataset
        k = DeNorm(data[10000].detach().cpu().numpy(), means["k_n_r"], stds["k_n_r"])
    else:
        data[10000] = (k.float() - means['k_n_r'].float())/stds['k_n_r'].float()
        
    ni_diff = DeNorm(item['n_i_difference'][idx].detach().cpu().numpy(), means['n_i_difference'], stds['n_i_difference'])
        
    prediction = model(data).detach().cpu().numpy()[0]
    prediction = DeNorm(prediction.detach().cpu().numpy(), means['n_i_difference'], stds['n_i_difference'])

    print(k) #the higher k, the smaller the ni peaks

    plt.plot(x,ni_diff,label='ni_diff')
    plt.plot(x,prediction,label='ni_diff prediction')
    
    return ni_diff, prediction, k


def get_Int_pred(x, dx, item, item_pred, idx, Amp=1, nr_mean = 0, nr_std = 1, ni_mean = 0, ni_std = 1):
    """
    Returns the Intensity of a given index in a batch of a dataloader.
    """
    
    k_nr = item['k_n_r'][idx]
    k = k_nr[0].cpu().numpy()
    k = DeNorm(k, nr_mean, nr_std)
    
    nr = k_nr[1:].cpu().numpy()
    nr = DeNorm(nr, nr_mean, nr_std)
    ni = item_pred[idx].detach().cpu().numpy()
    ni = DeNorm(ni, ni_mean, ni_std)
    
    n = nr + 1j*ni
    
    Int = plot_psi(x, dx, n, 1, k, plot=False)
    
    return Int
