import numpy as np
import matplotlib.pyplot as plt
import time


def numerov(x, dx, n, psi_r0, psi_r1, psi_i0, psi_i1, k):
    """
    Implementation of the numerov algorithm for the helmholtz equation. Needs two initial values and returns the complex wave as a numpy array.
    """

    psi = np.zeros(len(x), dtype=complex)
    psi[0] = complex(psi_r0, psi_i0)  # Startwerte
    psi[1] = complex(psi_r1, psi_i1)
    s = dx ** 2 / 12 * k ** 2

    # print('k: ', k)
    # print('psi[:1]', psi[:1])
    # print('s: ', s)

    for i in range(len(x) - 2):
        psi[i + 2] = (2 * (1 - 5 * s * n[i + 1] ** 2) * psi[i + 1] - (1 + s * n[i] ** 2) * psi[i])
        #print('psi[i + 2]: ', psi[i + 2])
        psi[i + 2] = psi[i + 2] / (1 + s * n[i + 2] ** 2)
        #print('psi[i + 2]: ', psi[i + 2])
        #break

    return psi


def init_val_in(x, dx, k, A):
    """
    Calculates a complex wave in empty space and returns the first two values. Needed as start values for the numerov algorithm. 
    """

    psi = np.zeros(2, dtype=complex)

    for i in range(len(psi)):
        psi[i] = A * np.cos(x[i] * k) + 1j * A * np.sin(x[i] * k)

    return psi[0].real, psi[1].real, psi[0].imag, psi[1].imag


def plot_psi(x, dx, n, Amp, k, plot=True):
    """
    Uses the numerov algorithm and the initial values from a wave in empty space to calculate psi and its intensity for wave propagation through a given refraction index. Plots the wave and returns the Intensity as numpy array.
    """

    psi_r0, psi_r1, psi_i0, psi_i1 = init_val_in(x, dx, k, Amp)

    nptime = time.time()
    psi = numerov(x, dx, n, psi_r0, psi_r1, psi_i0, psi_i1, k)
    nptime = time.time() - nptime

    f = len(x)

    Int = psi.real ** 2 + psi.imag ** 2

    if plot:
        plt.plot(x[-f:], psi.real[-f:], label='psi real')
        plt.plot(x[-f:], psi.imag[-f:], label='psi imag')

        plt.plot(x[-f:], n.real[-f:], label='nr')
        plt.plot(x[-f:], n.imag[-f:], label='ni')

        plt.plot(x[-f:], Int[-f:], label='Intensity')

        plt.legend(loc='upper left')

    return Int, nptime, psi.real, psi.imag


def get_psi(args, n, Amp, k):
    """
    Uses the numerov algorithm and the initial values from a wave in empty space to calculate psi and its intensity for wave propagation through a given refraction index. Plots the wave and returns the Intensity as numpy array.
    """

    psi_r0, psi_r1, psi_i0, psi_i1 = init_val_in(args.grid, args.stepsize, k, Amp)

    psi = numerov(args.grid,
                  args.stepsize,
                  n, psi_r0, psi_r1, psi_i0, psi_i1, k)

    Int = psi.real ** 2 + psi.imag ** 2

    return Int, psi.real, psi.imag
