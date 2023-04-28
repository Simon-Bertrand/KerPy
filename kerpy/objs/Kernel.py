import numpy as np
from functools import wraps 
class Kernel:
    def __init__(self, content):
        self.numpy = content
    def __repr__(self) -> str:
        return f"<kerpy.Kernel numpy = {self.numpy.__str__()} at {hex(id(self))}>"

    def to_reals(self):
        if self.numpy.dtype in [np.complex64, np.complex128]: return {key : f(self.numpy) for f,key in ((np.real, "x"), (np.imag, "y"))}
        else: return {"x":self.numpy}

    def pad(self, top=0, right=0, bot=0, left=0, value=0):
        paddings =  [top,-right,-bot,left]
        ker = value*np.ones((self.numpy.shape[0] + paddings[0] - paddings[2], self.numpy.shape[0] - paddings[1] + paddings[3]), dtype=self.numpy.dtype)
        paddings = list(map(lambda x: x if x!=0 else None, paddings))
        ker[paddings[0]:paddings[2], paddings[3]:paddings[1]] = self.numpy
        self.numpy = ker
        return self
    
    def rot90(self):
        self.numpy = np.rot90(self.numpy)
        return self
    
    def flip(self):
        self.numpy = np.flip(self.numpy)
        return self



    @staticmethod
    def decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            return Kernel( func(*args, **kwargs))
        return wrap