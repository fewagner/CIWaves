import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from lib.ComplexPyTorch import Complex


def PyNumerovReal(k_nr, ni, iv, args):
    """
    Computes the Numerov for a whole batch
    :param k_nr: a tensor of size [batchsize, arraylen +1] with nr and first value is k
    :param ni: a tensor of size [batchsize, arraylen +1] with ni
    :param h: stepsize
    :param iv: A dictionary with the first and second values of psir and psii, depending on k:
        {1: [psir0,psir1,psii0,psii1], 2: ...}
    :return: psir and psii in size [batchsize, arraylen] as tensors
    """
    k = torch.round(k_nr[:, 0]).double()
    nr = k_nr[:, 1:].double()
    ni = ni.double()
    s = args.stepsize ** 2 / 12

    psir = torch.zeros(nr.size()).to(args.device).double()
    psii = torch.zeros(nr.size()).to(args.device).double()

    # filling up initial values
    for i in range(len(psir)):
        psir[i, 0] = iv[int(k[i])][0]
        psir[i, 1] = iv[int(k[i])][1]
        psii[i, 0] = iv[int(k[i])][2]
        psii[i, 1] = iv[int(k[i])][3]

    # print('IV: ', psir[0, :2])

    # doing the numerov calculation
    for i in range(len(nr[0]) - 2):
        Vr2 = - (nr[:, i + 2] ** 2 - ni[:, i + 2] ** 2) * k ** 2
        Vr1 = - (nr[:, i + 1] ** 2 - ni[:, i + 1] ** 2) * k ** 2
        Vr0 = - (nr[:, i] ** 2 - ni[:, i] ** 2) * k ** 2
        Vi2 = - 2 * nr[:, i + 2] * ni[:, i + 2] * k ** 2
        Vi1 = - 2 * nr[:, i + 1] * ni[:, i + 1] * k ** 2
        Vi0 = - 2 * nr[:, i] * ni[:, i] * k ** 2

        a = 1 - s * Vr2
        b = s * Vi2
        c = s * Vr0 - 1
        d = - s * Vi0
        e = 10 * s * Vr1 + 2
        f = - 10 * s * Vi1

        # shit gettin real
        psir[:, i + 2] = (psir[:, i] * (a * c + b * d) + psir[:, i + 1] * (a * e + b * f) +
                          psii[:, i] * (a * d - b * c) + psii[:, i + 1] * (a * f - b * e)) \
                         / (a ** 2 + b ** 2)

        psii[:, i + 2] = (psir[:, i] * (- a * d - b * c) + psir[:, i + 1] * (- a * f - b * e) +
                          psii[:, i] * (a * c + b * d) + psii[:, i + 1] * (a * e - b * f)) \
                         / (a ** 2 + b ** 2)

    return psir, psii


def PyNumerovStepReal(k, nr2, nr1, nr0, ni2, ni1, ni0, psir1, psir0, psii1, psii0, args):
    """
    Computes the Numerov for a whole batch but just one step
    all inputs should be data type double!
    all inputs should be slices from the tensor
    """
    # k = torch.round(k).double()
    # nr2, nr1, nr0 = nr2.double(), nr1.double(), nr0.double()
    # ni2, ni1, ni0 = ni2.double(), ni1.double(), ni0.double()
    # psir1, psir0 = psir1.double(), psir0.double()
    # psii1, psii0 = psii1.double(), psii0.double()
    s = args.stepsize ** 2 / 12

    # doing the numerov step
    Vr2 = - (nr2 ** 2 - ni2 ** 2) * k ** 2
    Vr1 = - (nr1 ** 2 - ni1 ** 2) * k ** 2
    Vr0 = - (nr0 ** 2 - ni0 ** 2) * k ** 2
    Vi2 = - 2 * nr2 * ni2 * k ** 2
    Vi1 = - 2 * nr1 * ni1 * k ** 2
    Vi0 = - 2 * nr0 * ni0 * k ** 2

    a = 1 - s * Vr2
    b = s * Vi2
    c = s * Vr0 - 1
    d = - s * Vi0
    e = 10 * s * Vr1 + 2
    f = - 10 * s * Vi1

    # shit gettin real
    psir2 = (psir0 * (a * c + b * d) + psir1 * (a * e + b * f) +
             psii0 * (a * d - b * c) + psii1 * (a * f - b * e)) \
            / (a ** 2 + b ** 2)

    psii2 = (psir0 * (- a * d - b * c) + psir1 * (- a * f - b * e) +
             psii0 * (a * c + b * d) + psii1 * (a * e - b * f)) \
            / (a ** 2 + b ** 2)

    return psir2, psii2


def numerov(y0, y1, V, U, h, device):
    """
    Implementation of Numerov Algorithm to solve equations of the Form
    y''(x) = U(x) + V(x)*y(x)
    This function uses the Complex() class from ComplexPyTorch!
    :param device:
    :param y0,y1: initial values for y, both type Complex
        with real and imag as 1D tensor with size (batchsize)
    :param V: 1D tensor, same dimension as nr
    :param U: 1D tensor, same dimension as nr
    :param h: stepsize of the grid
    :return: y the solution as type Complex
    """

    s = Complex((h ** 2 / 12) * torch.ones(2 * V.size(0))).to(device)
    # print('s: ', s)
    # print('s.size(): ', s.size())
    y = Complex(torch.zeros(V.size(0), 2 * V.size(1))).to(device)

    y[:, 0] = y0.to(device)
    y[:, 1] = y1.to(device)

    F = U + V * y

    for i in range(V.size(1) - 2):
        # print('i: ',i)
        # print('y[:, i]: ', y[:, i])
        # print('y[:, i+1]: ', y[:, i+1])
        # print('U: ', U)
        # print('U.size():', U.size())
        # print('V: ', V)
        # print('V.size():', V.size())
        # print('V[:, i] * y[:, i]: ', (V[:, i] * y[:, i]).real)
        # print('1 - s * V[:, i + 2]: ', 1 - s * V[:, i + 2])
        # print('s * V[:, i + 2]: ', s * V[:, i + 2])
        y[:, i + 2] = (2 * y[:, i + 1] -
                       y[:, i] +
                       s * (U[:, i + 2] +
                            10 * F[:, i + 1] +
                            F[:, i])) / \
                      (1 - s * V[:, i + 2])
    return y


class CIDataset(Dataset):
    """
    Reads in the data with psi and forms it into a Dataset-Class for PyTorch.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.k_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.k_frame)

    def __getitem__(self, idx):
        nr = np.load(self.root_dir + 'n_real_' + str(idx) + '.npy')
        ni = np.load(self.root_dir + 'n_imag_' + str(idx) + '.npy')
        psir = np.load(self.root_dir + 'psir_' + str(idx) + '.npy')
        psii = np.load(self.root_dir + 'psii_' + str(idx) + '.npy')
        psir_hermitian = np.load(self.root_dir + 'psir_hermitean_' + str(idx) + '.npy')
        psii_hermitian = np.load(self.root_dir + 'psii_hermitean_' + str(idx) + '.npy')
        k = self.k_frame['k'][idx]

        k_nr = np.append(k, nr)

        sample = {'k_n_r': k_nr,
                  'n_i': ni,
                  'psir': psir,
                  'psii': psii,
                  'psir_hermitian': psir_hermitian,
                  'psii_hermitian': psii_hermitian
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample
