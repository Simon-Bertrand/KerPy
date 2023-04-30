"""Generate differential kernels"""
import numpy as np
from kerpy import shapes

from kerpy.objs.Kernel import Kernel


@Kernel.decorator
def prewitt(orient_x : bool =0, orient_y : bool=0, size : tuple[int, int]=(3,3)) -> Kernel:
    r"""Returns a complex kernel corresponding to the central finite difference 
    with multiple lines, also called Prewitt operator.
    The signs depend on the orient_x and orient_y args. 
    Interesting to compute an image gradient with a lower noise.

    Example : 
        .. math::
            \left(\begin{array}{ c, c, c}
            -1 + -1j  &  0 + -1j  &  1 + -1j \\
            -1 + 0j  &  0 + 0j  &  1 + 0j \\
            -1 + 1j  &  0 + 1j  &  1 + 1j 
            \end{array}\right)

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

def sobel(orient_x : bool =0, orient_y : bool=0, size : tuple[int, int]=(3,3)) -> Kernel:
    r"""Returns a complex kernel corresponding to the Sobel operator, 
    which is Prewitt plus the finite central.
    The signs depend on the orient_x and orient_y args. 
    Interesting to compute an image gradient with a lower noise.

    Example : 
        .. math::
            \left(
            \begin{array}{ c, c, c}
            -1 + -1j  &  0 + -2j  &  1 + -1j \\
            -2 + 0j  &  0 + 0j  &  2 + 0j \\
            -1 + 1j  &  0 + 2j  &  1 + 1j 
            \end{array}
            \right)

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
    return (prewitt(orient_x, orient_y, size) + central(orient_x, orient_y, size))


@Kernel.decorator
def robert_cross(orient_x : bool =0, orient_y : bool=0) -> Kernel:
    r"""Returns a 2x2 complex kernel corresponding to the Robert cross operator.
    The signs depend on the orient_x and orient_y args. 
    Interesting to compute an image gradient with a higher noise.

    Example : 
        .. math::
            \left(
            \begin{array}{ c, c}
            -1 + 0j  &  -0 + -1j \\
            0 + 1j  &  1 + 0j 
            \end{array}
            \right)

    :param orient_x: Goes forward in the x+y direction 
        if 0 else goes backwards, defaults to 0
    :type orient_x: 1|0
    :param orient_y: Goes forward in the x-y direction 
        if 0 else goes backwards, defaults to 0
    :type orient_y: 1|0
    :raises AssertionError: orient must belongs {0,1}
    :return: a Kernel object
    :rtype: Kernel
    """
    assert orient_x in [0,1] and orient_y in [0,1], "orient must belongs {0,1}"
    return np.array([[1*(2*orient_x-1),1j*(2*orient_y-1)], [-1j*(2*orient_y-1),-1*(2*orient_x-1)]])



@Kernel.decorator
def central(orient_x : bool =0, orient_y : bool=0, size : tuple[int, int]=(3,3)) -> Kernel:
    r"""Returns a complex kernel corresponding to the central finite difference.
    The signs depend on the orient_x and orient_y args. 
    Interesting to compute an image gradient.

    Example : 
        .. math::
            \left(\begin{array}{ c, c, c}
            0 + 0j  &  0 + -1j  &  0 + 0j \\
            -1 + 0j  &  0 + 0j  &  1 + 0j \\
            0 + 0j  &  0 + 1j  &  0 + 0j 
            \end{array}\right)

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
    ker[x_c, y_c-1] += 1 * (2*orient_x-1)
    ker[x_c, y_c+1] += -1 * (2*orient_x-1)
    ker[x_c-1, y_c] += 1j * (2*orient_y-1)
    ker[x_c+1, y_c] += -1j * (2*orient_y-1)

    return ker

@Kernel.decorator
def finite(orient_x : bool = 0, orient_y : bool= 0, flip: bool=True) -> Kernel:
    r"""Returns a complex 3x3 kernel that corresponds to finite forward or backward difference. 
    Interesting to compute an image gradient using the finite difference.

    Examples : 
        .. math::
            \left(\begin{array}{ c, c, c}
            0 + 0j  &  0 + -1j  &  0 + 0j \\
            -1 + 0j  &  1 + 1j  &  0 + 0j \\
            0 + 0j  &  0 + 0j  &  0 + 0j\end{array}\right)

    :param orient_x: Goes forward in the x direction if 0 else goes backward, defaults to 0
    :type orient_x: 1|0
    :param orient_y: Goes forward in the y direction if 0 else goes backward, defaults to 0
    :type orient_y: 1|0
    :param flip: Flip the kernel, defaults to True
    :type flip: boolean
    :raises AssertionError: orient must belongs {0,1}
    :return: a complex Kernel object
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


def laplacian(mode="diamond") -> Kernel:
    r"""Return a real kernel corresponding to the Laplacian kernel. 
    Interesting to compute an image Laplacian.

    Examples : 
        .. math::
            \left(\begin{array}{ c, c, c}
            0  & -1  &  0 \\
            -1  &  4  & -1 \\
            0  & -1  &  0 
            \end{array}\right)

    :param mode: Set the variant method for the Laplacian operator
    :type mode: "diamond"|"square"
    :return: a real Kernel object
    :rtype: Kernel
    """

    if mode =="diamond":
        ker = shapes.diamond(size=(3,3),scale=(1,1))
    elif mode == "square":
        ker = shapes.square(size=(3,3),scale=(1,1))
    else : raise ValueError("Mode must be in ['square', 'diamond']")
    ker.numpy[1,1] = -np.sum(ker.numpy) +1
    return ker * -1
