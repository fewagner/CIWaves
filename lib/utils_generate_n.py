import numpy as np
import matplotlib.pyplot as plt


def n_r(W, dW, x, k):
    """
    Calculates the real part of the refraction index n and returnes it as numpy array.
    """
    
    n_r = np.zeros(len(x))
    for i in range(len(x)):
        n_r[i] = W[i]/np.sqrt(2)*np.sqrt(1+np.sqrt(1+ dW[i]**2/(W[i]**4*k**2) ))
    
    return n_r

def n_i(dW, n_r, x, k):
    """
    Calculates the imaginary part of the refraction index n and returnes it as numpy array.
    """
    
    n_i = np.zeros(len(x))
    for i in range(len(x)):
        n_i[i] = -1/2/k*dW[i]/n_r[i]
    
    return n_i

def n_compl(W,dW,x,k):
    """
    Calculates the refraction index n as a complex number and returnes it as numpy array.
    """
    
    nr = n_r(W, dW, x, k)
    ni = n_i(dW, nr, x, k)
    n = nr + ni * 1j
    
    return n

def plot_n(x,W,dW,k):
    """
    Plots the real and imaginary part of n for given W and wave number k.
    """
    
    n = n_compl(W,dW,x,k)
    plt.plot(x,n.real,label='n_r')
    plt.plot(x,n.imag,label='n_i')
    plt.legend(loc='upper left')
    plt.show()