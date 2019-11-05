import torch
from collections.abc import Sequence


class Complex:
    """Complex tensors for PyTorch"""

    def __init__(self, real, **imag):
        """
        the inputs can either be tensors and have same dimension (named real and imag) or
        one tensor with even length in last dimension, then first half real, second half imag
        Example usage:
        compl_tensor = Complex(real=xr, imag=xi)
        compl_tensor = Complex(x)
        """
        if ('imag' in imag):
            self.real = real.double()
            self.imag = imag['imag'].double()
        elif real.size(-1) % 2 == 0:
            self.real = real[..., :int(real.size(-1) / 2)].double()
            self.imag = real[..., int(real.size(-1) / 2):].double()
        else:
            print('INITIALIZATION FAILED!')
        super().__init__()

    # Stuff that changes the object
    def __setitem__(self, i, x):  # x must be a Complex tensor itself
        self.real[i] = x.real
        self.imag[i] = x.imag

    def add(self, x):
        """argument has to be Class Complex too! or int"""
        if type(x) is int:
            self.real += x
        else:
            self.real = self.real + x.real
            self.imag = self.imag + x.imag

    def mul(self, x):
        if type(x) is int:
            self.real *= x
            self.imag *= x
        else:
            a = self.real
            b = self.imag
            self.real = a * x.real - b * x.imag
            self.imag = b * x.real + a * x.imag

    def div(self, x):
        if type(x) is int:
            self.real /= x
            self.imag /= x
        else:
            a = self.real
            b = self.imag
            self.real = (a * x.real + b * x.imag) / (x.real ** 2 + x.imag ** 2)
            self.imag = (b * x.real - a * x.imag) / (x.real ** 2 + x.imag ** 2)

    # Stuff that returns something
    def __getitem__(self, i):
        return Complex(real=self.real[i], imag=self.imag[i])

    def __len__(self):
        return len(self.real)

    def view(self, *args):
        return Complex(real=self.real.view(args), imag=self.imag.view(args))

    def size(self, *i):
        if len(i) == 0:
            return self.real.size()
        if len(i) == 1:
            return self.real.size(i[0])
        else:
            print('IGNORING SIZE ARGUMENT, MUST BE INT!')
            return self.real.size()

    def __neg__(self):
        return Complex(real=- self.real, imag=- self.imag)

    def __add__(self, x):
        if isinstance(x, (int, float)):
            return Complex(real=self.real + x, imag=self.imag)
        else:
            return Complex(real=self.real + x.real, imag=self.imag + x.imag)

    def __radd__(self, x):
        if isinstance(x, (int, float)):
            return Complex(real=x + self.real, imag=self.imag)
        else:
            return Complex(real=x.real + self.real, imag=x.imag + self.imag)

    def __sub__(self, x):
        if isinstance(x, (int, float)):
            return Complex(real=self.real - x, imag=self.imag)
        else:
            return Complex(real=self.real - x.real, imag=self.imag - x.imag)

    def __rsub__(self, x):
        if isinstance(x, (int, float)):
            return Complex(real=x - self.real, imag= - self.imag)
        else:
            return Complex(real=x.real - self.real, imag=x.imag - self.imag)

    def __mul__(self, x):
        if type(x) is (int,float):
            return Complex(real=self.real * x, imag=self.imag * x)
        else:
            return Complex(real=self.real * x.real - self.imag * x.imag,
                           imag=self.imag * x.real + self.real * x.imag)

    def __rmul__(self, x):
        if type(x) is (int,float):
            return Complex(real=x * self.real, imag=x * self.imag)
        else:
            return Complex(real=self.real * x.real - self.imag * x.imag,
                           imag=self.imag * x.real + self.real * x.imag)

    def __truediv__(self, x):
        if type(x) is (int,float):
            return Complex(real=self.real / x, imag=self.imag / x)
        else:
            return Complex(real=(self.real * x.real + self.imag * x.imag) / (x.real ** 2 + x.imag ** 2),
                           imag=(self.imag * x.real - self.real * x.imag) / (x.real ** 2 + x.imag ** 2))

    def __str__(self):
        if not self.dim() == 0:
            return str(torch.cat((self.real, self.imag), dim=-1))
        else:
            return str([self.real, self.imag])

    def __pow__(self, x):
        return Complex(real=self.abs() ** x * torch.cos(x * self.phi()),
                       imag=self.abs() ** x * torch.sin(x * self.phi()))

    def val(self):
        return torch.cat((self.real, self.imag), dim=-1)

    def conj(self):
        return torch.cat((self.real, -self.imag), dim=-1)

    def abs(self):
        return torch.sqrt(self.real ** 2 + self.imag ** 2)

    def phi(self):
        return torch.atan(self.imag / self.real)

    def to(self, device):
        return Complex(real=self.real.to(device),
                       imag=self.imag.to(device))

    def dim(self):
        return self.real.dim()