"""Generate image processing kernels"""
import numpy as np
from kerpy.diff import laplacian
from kerpy.objs.Kernel import Kernel

@Kernel.decorator
def gaussian(size=(3,3), std=(1,1), normalize=True) -> Kernel:
    """Returns a real kernel that corresponds to the
    gaussian kernel.
    :param size: Tuple defining the size of the kernel respectively
    in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :param std: Tuple defining the standard deviation of the kernel
    respectively in the x and y direction, defaults to (1,1)
    :type std: (int, int)
    :param normalize: Set the sum of all coefficients equal to one
    to conserve to global mass.
    :type normalize: bool
    :return: a Kernel object
    :rtype: Kernel
    """
    x_arr,y_arr = np.meshgrid(np.arange(0,size[0]),np.arange(0,size[1]))
    x_c = (size[0]-1)//2
    y_c = (size[1]-1)//2
    ker = np.exp(-(((x_arr-x_c)**2)/(2*std[0]**2) + ((y_arr-y_c)**2)/(2*std[1]**2)))
    return ker/np.sum(ker) if normalize else ker

@Kernel.decorator
def mean(size=(3,3)) -> Kernel:
    """Returns a real kernel that corresponds to the ones matrix
    divided the number of element.
    :param size: Tuple defining the size of the kernel respectively
    in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    return np.ones(size)/np.prod(size)

def sharpen(size : tuple[int, int]=(3,3)) -> Kernel:
    """Returns the identitical kernel plus the laplacian one in
    order to sharpen an image.
    :param size: Tuple defining the size of the kernel respectively
    in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    ker : Kernel = laplacian(size)
    # pylint: disable=no-member
    ker.numpy[(size[0]-1)//2,(size[1]-1)//2] +=1
    return ker

def unsharp(size=(3,3), std=(1,1)) -> Kernel:
    """Returns a real kernel that corresponds to the gaussian kernel.
    :param size: Tuple defining the size of the kernel respectively
    in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :param std: Tuple defining the standard deviation of the kernel
    respectively in the x and y direction, defaults to (1,1)
    :type std: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    ker : Kernel = gaussian(size, std, True)
    ker.numpy[(size[0]-1)//2,(size[1]-1)//2] -= 2
    ker.numpy *= -1
    return ker
