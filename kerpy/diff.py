"""Generate differential kernels"""
import numpy as np

from kerpy.objs.Kernel import Kernel


@Kernel.decorator
def lines(orient_x : bool =0, orient_y : bool=0, size : tuple[int, int]=(3,3)) -> Kernel:
    """Returns a complex kernel with a line of signed
    one after the center and a line of opposite signed
    one before the center.The signs depend on the orient_x
    and orient_y args. Interesting to compute an image
    gradient with a lower noise.

    :param orient_x: Goes forward in the x direction 
    if 0 else goes backwards, defaults to 0
    :type orient_x: 1|0
    :param orient_y: Goes forward in the y direction 
    if 0 else goes backwards, defaults to 0
    :type orient_y: 1|0
    :param size: Tuple defining the size of the kernel 
    respectively in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :raises AssertionError: orient must belongs {0,1}
    :return: a Kernel object
    :rtype: Kernel
    """
    assert orient_x in [0,1] and orient_y in [0,1], "orient must belongs {0,1}"
    ker = np.zeros(size, dtype=np.complex64)
    x_c,y_c =(size[0]-1)//2, (size[1]-1)//2
    ker[:, y_c-1] += 1 * (2*orient_x-1)
    ker[:, y_c+1] += -1 * (2*orient_x-1)
    ker[x_c-1, :] += 1j * (2*orient_y-1)
    ker[x_c+1, :] += -1j * (2*orient_y-1)

    return ker

@Kernel.decorator
def dual(orient_x : bool = 0, orient_y : bool= 0, flip: bool=True) -> Kernel:
    """Returns a complex 3x3 kernel that compute the first order finite difference. 
    The signs depend on the orient_x and orient_y args. 
    Interesting to compute an image gradient using the differential of the finite order.
    
    :param orient_x: Goes forward in the x direction if 0 else goes backward, defaults to 0
    :type orient_x: 1|0
    :param orient_y: Goes forward in the y direction if 0 else goes backward, defaults to 0
    :type orient_y: 1|0
    :param flip: Flip the kernel, defaults to True
    :type flip: boolean
    :raises AssertionError: orient must belongs {0,1}
    :return: a Kernel object
    :rtype: Kernel
    """
    assert orient_x in [0,1] and orient_y in [0,1], "orient must belongs {0,1}"
    ker = np.zeros((3, 3), dtype=np.complex64)
    ker[1,1] = -1*(2*orient_x-1) -1j * (2*orient_y-1)
    ker[2,1] += 1j*(2*orient_y-1)
    ker[1,2] += 1 * (2*orient_x-1)
    if flip:
        return np.flip(ker)
    return ker

@Kernel.decorator
def laplacian(size : tuple[int, int]=(3,3)) -> Kernel:
    """Return a real kernel corresponding to the Laplacian kernel. 
    Interesting to compute an image Laplacian.
    :param size: Tuple defining the size of the kernel respectively
    in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :return: a complex Kernel object
    :rtype: Kernel
    """
    ker = np.zeros(size)
    x_c = (size[0]-1)//2
    y_c = (size[1]-1)//2
    ker[y_c,:] = -1
    ker[:,x_c] = -1
    ker[x_c,y_c] = -np.sum(ker) -1
    return ker
