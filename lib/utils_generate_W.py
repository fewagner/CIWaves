import numpy as np


def get_grid(half_width, nmbr_points):
    """
    Calculates a Grid with nmbr_points points in the range [-half_wifth, half_width] and returns in as numpy array, together with the stepsize. 
    """
    
    grid = np.arange(-half_width, half_width, 2*half_width/nmbr_points)
    stepsize = grid[1] - grid[0]
    
    return grid, stepsize


def gauss_dist(mu, sig, x): 
    """
    Returns the value of a Gauss distribution with mean mu and std sig at point x.
    """
    
    return np.exp(-(x-mu)**2/(2*sig**2))/(2*np.pi*sig**2)


def generate_W(x):
    """
    Generates a W which is needed to calculate constant intensity waves. Returns is as numpy array of the same size as the grid x is.
    """
    
    rate = 0.3
    ran = np.zeros(len(x))
    asym = int(len(ran)/4)
    for i in range(asym, len(ran) - asym):
        coin = np.random.uniform()
        if coin < rate:
            ran[i] = np.random.uniform() / (0.4 *asym)
    con = np.convolve(ran, gauss_dist(0,0.1,x)) + 1
    a = int(len(x)/2)
    
    return con[a:a+len(x)]


def generate_W_diverse(x):
    """
    Generates a W which is needed to calculate constant intensity waves. Returns is as numpy array of the same size as the grid x is.
    different sigma values are possible
    """
    
    Ws = []
    
    rate = 0.3
    ran = np.zeros(len(x))
    asym = int(len(ran)/4)
    for i in range(asym, len(ran) - asym):
        coin = np.random.uniform()
        if coin < rate:
            ran[i] = np.random.uniform() / (0.4 *asym)
    sigmas = np.random.uniform(0.05,0.15,3)
    for sigma in sigmas:
        con = np.convolve(ran, gauss_dist(0,sigma,x)) + 1
        a = int(len(x)/2)
        Ws.append(con[a:a+len(x)])
    
    return Ws