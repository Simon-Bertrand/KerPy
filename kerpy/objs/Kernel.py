import numpy as np
from functools import wraps 
class Kernel:
    def __init__(self, content):
        if not isinstance(content, np.ndarray):
            raise ValueError("Only nd.array are allowed in the Kernel constructor")
        self.numpy = content
    def __repr__(self) -> str:
        return f"<kerpy.Kernel numpy =\n{self.numpy.__str__()} at {hex(id(self))}>"

    def to_reals(self):
        r"""Compute the real and imaginary part of a complex Kernel

        :return: a dict of real Kernel object
        :rtype: dict
        """
        if self.numpy.dtype in [np.complex64, np.complex128]: return {key : f(self.numpy) for f,key in ((np.real, "x"), (np.imag, "y"))}
        else: return {"x":self.numpy}

    def divergence(self):
        r"""Compute the divergence kernel of a complex kernel

        :return: a real Kernel object
        :rtype: Kernel
        """
        if self.numpy.dtype in [np.complex64, np.complex128]: 
            self.numpy = np.sum([f(self.numpy) for f in (np.real, np.imag)], axis=0)
        return self

    def pad(self, top=0, right=0, bot=0, left=0, value=0):
        r"""Add padding to a Kernel

        :param top: Adds top padding
        :type top: int
        :param right: Adds right padding
        :type right: int
        :param bot: Adds bot padding
        :type bot: int
        :param left: Adds left padding
        :type left: int
        :param value: Padding value
        :type value: int
        :return: a Kernel object
        :rtype: Kernel
        """
        paddings =  [top,-right,-bot,left]
        ker = value*np.ones((self.numpy.shape[0] + paddings[0] - paddings[2], self.numpy.shape[0] - paddings[1] + paddings[3]), dtype=self.numpy.dtype)
        paddings = list(map(lambda x: x if x!=0 else None, paddings))
        ker[paddings[0]:paddings[2], paddings[3]:paddings[1]] = self.numpy
        self.numpy = ker
        return self
    
    def stride(self, strides=1, value=0):
        r"""Add regularly new constant value between coefficients

        :param strides: Margin width to add between coefficients
        :type strides: int
        :param value: Strided value
        :type value: int
        :return: a Kernel object
        :rtype: Kernel
        """
        ker = self.numpy
        strided_ker = value*np.ones((np.array(ker.shape))*(strides+1) + strides)
        for x in range(ker.shape[0]):
            for y in range(ker.shape[1]):
                strided_ker[(strides+1)*x + strides, (strides+1)*y + strides] = ker[x,y]

        self.numpy = strided_ker
        return self
    
    def rot90(self):
        r"""Rotate by 90 degrees

        :return: a Kernel object
        :rtype: Kernel
        """
        self.numpy = np.rot90(self.numpy)
        return self
    
    def flip(self):
        r"""Flip a kernel

        :return: a Kernel object
        :rtype: Kernel
        """
        self.numpy = np.flip(self.numpy)
        return self
    

    def __add__(self, other): return Kernel(self.numpy.__add__(other.numpy if isinstance(other, Kernel) else other))
    def __sub__(self, other): return Kernel(self.numpy.__sub__(other.numpy if isinstance(other, Kernel) else other) )
    def __mul__(self, other): return Kernel(self.numpy.__mul__(other.numpy if isinstance(other, Kernel) else other) )
    def __truediv__(self, other): return Kernel(self.numpy.__truediv__(other.numpy if isinstance(other, Kernel) else other) )
    def __floordiv__(self, other): return Kernel(self.numpy.__floordiv__(other.numpy if isinstance(other, Kernel) else other) )
    def __mod__(self, other): return Kernel(self.numpy.__mod__(other.numpy if isinstance(other, Kernel) else other) )
    def __pow__(self, other): return Kernel(self.numpy.__pow__(other.numpy if isinstance(other, Kernel) else other) )
    def __rshift__(self, other): return Kernel(self.numpy.__rshift__(other.numpy if isinstance(other, Kernel) else other) )
    def __lshift__(self, other): return Kernel(self.numpy.__lshift__(other.numpy if isinstance(other, Kernel) else other) )
    def __and__(self, other): return Kernel(self.numpy.__and__(other.numpy if isinstance(other, Kernel) else other) )
    def __or__(self, other): return Kernel(self.numpy.__or__(other.numpy if isinstance(other, Kernel) else other) )
    def __xor__(self, other): return Kernel(self.numpy.__xor__(other.numpy if isinstance(other, Kernel) else other) )
    def __lt__(self, other): return Kernel(self.numpy.__lt__ (other.numpy if isinstance(other, Kernel) else other) )
    def __gt__(self, other): return Kernel(self.numpy.__gt__ (other.numpy if isinstance(other, Kernel) else other) )
    def __le__(self, other): return Kernel(self.numpy.__le__(other.numpy if isinstance(other, Kernel) else other) )
    def __ge__(self, other): return Kernel(self.numpy.__ge__(other.numpy if isinstance(other, Kernel) else other) )
    def __eq__(self, other): return Kernel(self.numpy.__eq__(other.numpy if isinstance(other, Kernel) else other) )
    def __ne__(self, other): return Kernel(self.numpy.__ne__(other.numpy if isinstance(other, Kernel) else other) )
    def ___invert__(self, other): return Kernel(self.numpy.___invert__(other.numpy if isinstance(other, Kernel) else other) )
 






    @staticmethod
    def decorator(func):
        """
        :meta private:
        """
        @wraps(func)
        def wrap(*args, **kwargs):
            return Kernel( func(*args, **kwargs))
        return wrap