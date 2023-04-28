import numpy as np

from kerpy.objs.Kernel import Kernel

@Kernel.decorator
def gaussian(size=(3,3), std=(1,1), normalize=True):
    X,Y = np.meshgrid(np.arange(0,size[0]),np.arange(0,size[1]))

    X0 = (size[0]-1)//2
    Y0 = (size[1]-1)//2
    
    ker = np.exp(-(((X-X0)**2)/(2*std[0]**2) + ((Y-Y0)**2)/(2*std[1]**2)))
    return ker/np.sum(ker) if normalize else ker