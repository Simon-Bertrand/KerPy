"""Generate image processing kernels"""
import numpy as np
from .diff import laplacian
from .objs.Kernel import Kernel

@Kernel.decorator
def gaussian(size=(3,3), std=(1,1), normalize=True) -> Kernel:
    r"""Returns a real kernel that corresponds to the
    gaussian kernel.

    Example : 
        .. math::
            \left(
            \begin{array}{ c, c, c}
            0.08 &  0.12 &  0.08\\
            0.12 &  0.20 &  0.12\\
            0.08 &  0.12 &  0.08
            \end{array}
            \right)

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
    r"""Returns a real kernel that corresponds to the ones matrix
    divided the number of element.

    Example : 
        .. math::
            \left(
            \begin{array}{ c, c, c}
            0.11 &  0.11 &  0.11\\
            0.11 &  0.11 &  0.11\\
            0.11 &  0.11 &  0.11
            \end{array}
            \right)

    :param size: Tuple defining the size of the kernel respectively
        in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    return np.ones(size)/np.prod(size)

def sharpen(size : tuple[int, int]=(3,3)) -> Kernel:
    r"""Returns the identitical kernel plus the laplacian in
    order to sharpen an image.

    Example : 
        .. math::
            \left(
            \begin{array}{ c, c, c}
            0.00 & -1.00 &  0.00\\
            -1.00 &  5.00 & -1.00\\
            0.00 & -1.00 &  0.00
            \end{array}
            \right)

    :param size: Tuple defining the size of the kernel respectively
        in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    ker : Kernel = laplacian()
    # pylint: disable=no-member
    ker.numpy[(size[0]-1)//2,(size[1]-1)//2] +=1
    return ker

def unsharp(size=(3,3), std=(1,1)) -> Kernel:
    r"""Returns a real kernel that corresponds to the gaussian kernel minus 
    two times the identity kernel.

    Example : 
        .. math::
            \left(
            \begin{array}{ c, c, c}
            -0.08 & -0.12 & -0.08\\
            -0.12 &  1.80 & -0.12\\
            -0.08 & -0.12 & -0.08
            \end{array}
            \right)

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
