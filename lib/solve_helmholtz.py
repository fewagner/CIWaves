import torch
from lib.ComplexPyTorch import Complex


def solve_helmholtz(nr, ni, iv, k, h, device):
    """
    Implementation of Numerov Algorithm to solve the Helmholtz Equation for different n, k
    This function uses the Complex() class from ComplexPyTorch and numerov!
    :param device:
    :param nr: 1D tensor
    :param ni: 1D tensor, same dimension as nr
    :param iv: A dictionary with the first and second values of psir and psii, depending on k:
                {1: [psir0,psir1,psii0,psii1], 2: ...}
    :param k: wave vector
    :param h: stepsize of the grid
    :return: two tensors psi real and psi imag
    """
    n = Complex(real=nr, imag=ni).to(device)
    k = Complex(real=k,
                imag=torch.zeros(k.size())).to(device)
    # V = - k ** 2 * n ** 2
    # U = Complex(torch.zeros(V.size(0), 2 * V.size(1))).to(device)
    y0 = Complex(torch.zeros(2 * len(k))).to(device)
    y1 = Complex(torch.zeros(2 * len(k))).to(device)

    # filling up initial values
    for i in range(len(k)):
        y0.real[i] = iv[int(k[i].real)][0]  # real part
        y1.real[i] = iv[int(k[i].real)][1]
        y0.imag[i] = iv[int(k[i].real)][2]  # imaginary part
        y1.imag[i] = iv[int(k[i].real)][3]

    y = Complex(torch.zeros(nr.size(0), 2 * nr.size(1))).to(device)
    s = Complex(real=h ** 2 / 12 * torch.ones(k.size()),
                imag=torch.zeros(k.size()))
    s = s * k ** 2

    y[:, 0] = y0.to(device)
    y[:, 1] = y1.to(device)
    #
    # print('k[0]: ', k[0])
    # print('psi[0,:1]', y[0,:1])
    # print('s[0]: ', s[0])

    for i in range(nr.size(1) - 2):
        y[:, i + 2] = (2 *
                       (1 - 5 * s * n[:, i + 1] ** 2) *
                       y[:, i + 1] -
                       (1 + s * n[:, i] ** 2) *
                       y[:, i])
        y[:, i + 2] = y[:, i + 2] / (
                              1 +
                              s *
                              n[:, i + 2] ** 2)

    return y.real, y.imag