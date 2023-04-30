"""Generate shapes kernels"""
import numpy as np

from .objs.Kernel import Kernel

def _shapes_generator(
        size=(3,3),
        scale = (1,1),
        condition = lambda x_c,y_c, scale : np.sqrt(scale[1]*x_c**2 + scale[0]*y_c**2)<1,
        mode="fill"
    ):
    x_arr,y_arr = np.meshgrid(np.arange(size[0]),np.arange(size[1]))
    if mode=="fill":
        return 1*condition(x_arr-(size[1]-1)//2, y_arr-(size[0]-1)//2, scale)
    if mode =="outline":
        return 1*(
            condition(x_arr-(size[1]-1)//2, y_arr-(size[0]-1)//2, scale)
            &
            ~(condition(x_arr-(size[1]-1)//2, y_arr-(size[0]-1)//2, (scale[0]-1, scale[1]-1)))
        )
    raise ValueError("Mode of shape must be in ['fill', 'outline']")

@Kernel.decorator
def circle(size=(21,21),scale=(1,1), mode="fill"):
    r"""Returns a circle kernel of given scale.
    
    Example : 
        .. math::
            \left(
            \begin{array}{ c, c, c, c, c}
            0  &  0  &  1  &  0  &  0 \\
            0  &  1  &  1  &  1  &  0 \\
            1  &  1  &  1  &  1  &  1 \\
            0  &  1  &  1  &  1  &  0 \\
            0  &  0  &  1  &  0  &  0 
            \end{array}
            \right)

    :param size: Tuple defining the size of the kernel respectively
        in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :param scale: Tuple defining the width and height of the circle defaults to (1,1)
    :type std: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    return _shapes_generator(
        size,scale,mode=mode,
        condition =
        lambda x_c,y_c,scale:(np.sqrt((x_c/(1+scale[0]))**2+(y_c/(1+scale[1]))**2<=1))
    )

@Kernel.decorator
def diamond(size=(21,21),scale=(1,1), mode="fill"):
    r"""Returns a diamond kernel of given scale.

    Example : 
        .. math::
            \left(
            \begin{array}{ c, c, c, c, c}
            0  &  0  &  1  &  0  &  0 \\
            0  &  1  &  1  &  1  &  0 \\
            1  &  1  &  1  &  1  &  1 \\
            0  &  1  &  1  &  1  &  0 \\
            0  &  0  &  1  &  0  &  0 
            \end{array}
            \right)

    :param size: Tuple defining the size of the kernel respectively
        in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :param scale: Tuple defining the width and height of the diamond defaults to (1,1)
    :type std: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    return _shapes_generator(
        size,scale, mode=mode,
        condition =
        lambda x_c,y_c, scale : (np.abs(x_c)/(1+scale[0]) + np.abs(y_c)/(1+scale[1]))<1
    )

@Kernel.decorator
def square(size=(21,21),scale=(1,1), mode="fill"):
    r"""Returns a square kernel of given scale.

    Example : 
        .. math::
            \left(\begin{array}{ c, c, c, c, c}
            0  &  0  &  0  &  0  &  0 \\
            0  &  1  &  1  &  1  &  0 \\
            0  &  1  &  1  &  1  &  0 \\
            0  &  1  &  1  &  1  &  0 \\
            0  &  0  &  0  &  0  &  0 
            \end{array}\right)

    :param size: Tuple defining the size of the kernel respectively
        in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :param scale: Tuple defining the width and height of the square defaults to (1,1)
    :type std: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    return _shapes_generator(
        size,scale, mode=mode,
        condition =
        lambda x_c,y_c,s:((y_c/s[0]<=1)&(-1<=y_c/s[0])&(-1<=x_c/s[1])&(x_c/s[1]<=1))
    )

@Kernel.decorator
def triangle(size=(21,21),scale=(1,1)):
    r"""Returns a triangle kernel of given scale.

    Example : 
        .. math::
            \left(
            \begin{array}{ c, c, c, c, c}
            0  &  0  &  0  &  0  &  0 \\
            0  &  0  &  1  &  0  &  0 \\
            0  &  1  &  1  &  1  &  0 \\
            1  &  1  &  1  &  1  &  1 \\
            0  &  0  &  0  &  0  &  0 
            \end{array}
            \right)

    :param size: Tuple defining the size of the kernel respectively
        in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :param scale: Tuple defining the width and height of the triangle defaults to (1,1)
    :type std: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    return _shapes_generator(
        size,scale,
        condition =
        lambda x_c,y_c,s:((y_c/s[0]< 2)&(2>x_c/s[1]-(y_c/s[0]))&(-2<y_c/s[0]+x_c/s[1]))
    )

@Kernel.decorator
def cross(size=(21,21),scale=(1,1)):
    r"""Returns a cross kernel of given scale.

    Example : 
        .. math::
            \left(\begin{array}{ c, c, c, c, c}
            0  &  0  &  1  &  0  &  0 \\
            0  &  0  &  1  &  0  &  0 \\
            1  &  1  &  1  &  1  &  1 \\
            0  &  0  &  1  &  0  &  0 \\
            0  &  0  &  1  &  0  &  0 
            \end{array}\right)

    :param size: Tuple defining the size of the kernel respectively
        in the x and y direction, defaults to (3,3)
    :type size: (int, int)
    :param scale: Tuple defining the width and height of the cross defaults to (1,1)
    :type std: (int, int)
    :return: a Kernel object
    :rtype: Kernel
    """
    return _shapes_generator(
        size,scale,
        condition =
        lambda x_c,y_c,s :((y_c==0)|(x_c==0)) &((y_c<=s[1])&(y_c>=-s[1])&(x_c<=s[0])&(x_c>=-s[1]))
    )
