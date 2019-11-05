import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import matplotlib as mpl


class Norm(object):
    """"Data ist then normalized to given mean and std."""

    def __init__(self, args):
        self.nr_mean = args.nr_mean
        self.nr_std = args.nr_std
        self.ni_mean = args.ni_mean
        self.ni_std = args.ni_std

    def __call__(self, sample):
        k_nr, ni = sample['k_n_r'], sample['n_i']

        k_nr = (k_nr - self.nr_mean) / self.nr_std
        ni = (ni - self.ni_mean) / self.ni_std

        return {'k_n_r': k_nr,
                'n_i': ni,
                'psir': sample['psir'],
                'psii': sample['psii'],
                'psir_hermitian': sample['psir_hermitian'],
                'psii_hermitian': sample['psii_hermitian']
                }


def DN(x, p, args):

    """
    Takes a normalized sample, together with mean and std to that it should be scaled back and returns the original sample.
    p = r means real part, = i means imaginary
    """
    if p == 'r':
        x = x * args.nr_std + args.nr_mean
    elif p == 'i':
        x = x * args.ni_std + args.ni_mean
    else:
        return 'error p is not r,i'

    return x


def plot_bars(bar_heights, bar_labels, path, path_here, nameOfPlot):
    plt.style.use('file://' + path_here + 'PaperDoubleFig.mplstyle')
    colourWheel = ['#329932',
                   '#ff6961',
                   'b',
                   '#6a3d9a',
                   '#fb9a99',
                   '#e31a1c',
                   '#fdbf6f',
                   '#ff7f00',
                   '#cab2d6',
                   '#6a3d9a',
                   '#ffff99',
                   '#b15928',
                   '#67001f',
                   '#b2182b',
                   '#d6604d',
                   '#f4a582',
                   '#fddbc7',
                   '#f7f7f7',
                   '#d1e5f0',
                   '#92c5de',
                   '#4393c3',
                   '#2166ac',
                   '#053061']
    dashesStyles = [[3, 1],
                    [1000, 1],
                    [2, 1, 10, 1],
                    [4, 1, 1, 1, 1, 1]]
    plt.close('all')
    fig, ax = plt.subplots()

    alphaVal = 0.6

    alignment = np.arange(len(bar_labels))
    bar_heights = np.around(bar_heights,
                            decimals=2)
    plt.bar(alignment,
            bar_heights,
            color=colourWheel[3 % len(colourWheel)],
            alpha = alphaVal)
    plt.xticks(alignment, bar_labels)
    max = np.max(bar_heights)

    d = 1.0
    for i in range(len(bar_heights)):
        if bar_heights[i] > 0.5 * max:
            plt.text(-0.3 + i * d, bar_heights[i] - 0.2 * max, bar_heights[i], ha='left', rotation=60, color='w')
        else:
            plt.text(-0.3 + i * d, bar_heights[i] + 0.02 * max, bar_heights[i], ha='left', rotation=60)

    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_label_coords(0.53, 1.01)
    ax.yaxis.tick_right()
    nameOfPlot = nameOfPlot
    plt.ylabel(nameOfPlot, rotation=0)

    fig.savefig(path, bbox_inches="tight", dpi=300)


def R(psi_xmin1, psi_x0, deltax, k):
    """
    Reflection Coefficient of the Wave on the potential
    :param psi_xmin1: the value of psi on the forelast node 
    :param psi_x0: the value of psi on the last node
    :param deltax: stepsize of the grid
    :param k: wave vector
    :return: a real Number R, the reflection coefficient
    """
    return np.abs((psi_xmin1 * np.exp(- 1j * k * deltax) - psi_x0)) ** 2 / \
           np.abs((psi_xmin1 * np.exp(+1j * k * deltax) - psi_x0)) ** 2


def T(psi_xmin1, psi_x0, deltax, k):
    """
    Transmission Coefficient of the Wave on the potential
    :param psi_xmin1: the value of psi on the forelast node
    :param psi_x0: the value of psi on the last node
    :param deltax: stepsize of the grid
    :param k: wave vector
    :return: a real Number T, the transmission coefficient
    """
    return np.abs(np.exp(2j * k * deltax) - 1) ** 2 / \
           np.abs(psi_xmin1 * np.exp(1j * k * deltax) - psi_x0) ** 2
