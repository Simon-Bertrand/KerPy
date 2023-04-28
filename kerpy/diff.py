import numpy as np

from kerpy.objs.Kernel import Kernel


@Kernel.decorator
def lines(orient_x, orient_y):
    """blah blah blah

    :meta public:
    """
    assert orient_x in [0,1] and orient_y in [0,1], "orient must belongs {0,1}"
    ker = np.zeros((3, 3), dtype=np.complex64)
    ker[:, 0] += 1 * (2*orient_x-1)
    ker[:, -1] += -1 * (2*orient_x-1)
    ker[0, :] += 1j * (2*orient_y-1)
    ker[-1, :] += -1j * (2*orient_y-1)

    return ker

@Kernel.decorator
def dual(orient_x, orient_y, flip=True):
    assert orient_x in [0,1] and orient_y in [0,1], "orient must belongs {0,1}"
    ker = np.zeros((3, 3), dtype=np.complex64)
    ker[1,1] = -1*(2*orient_x-1) -1j * (2*orient_y-1)
    ker[2,1] += 1j*(2*orient_y-1)
    ker[1,2] += 1 * (2*orient_x-1)
    
    if flip: return np.flip(ker)
    else: return ker

@Kernel.decorator        
def laplacian(size=(3,3)):
    ker = np.zeros(size)
    X0 = (size[0]-1)//2
    Y0 = (size[1]-1)//2
    ker[Y0,:] = 1
    ker[:,X0] = 1
    ker[X0,Y0] = -np.sum(ker) +1
    return ker
