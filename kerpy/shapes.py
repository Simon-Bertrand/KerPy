import numpy as np

from kerpy.objs.Kernel import Kernel

def _shapes_generator(size=(3,3), scale = (1,1), condition = lambda Xc,Yc, scale : np.sqrt(scale[1]*Xc**2 + scale[0]*Yc**2)<1, mode="fill"):
    X,Y = np.meshgrid(np.arange(size[0]),np.arange(size[1]))
    if mode=="fill":
        return 1*condition(X-(size[1]-1)//2, Y-(size[0]-1)//2, scale)
    elif mode =="outline":
        return 1*(condition(X-(size[1]-1)//2, Y-(size[0]-1)//2, scale) & ~(condition(X-(size[1]-1)//2, Y-(size[0]-1)//2, (scale[0]-1, scale[1]-1))))
    else: raise ValueError("Mode of shape must be in ['fill', 'outline']")

@Kernel.decorator
def circle(size=(21,21),scale=(1,1), mode="fill"): return _shapes_generator(size,scale, mode=mode, condition = lambda Xc,Yc, scale : (np.sqrt((Xc/(1+scale[0]))**2 + (Yc/(1+scale[1]))**2)<1))

@Kernel.decorator
def diamond(size=(21,21),scale=(1,1), mode="fill"): return _shapes_generator(size,scale, mode=mode, condition = lambda Xc,Yc, scale : (np.abs(Xc)/(1+scale[0]) + np.abs(Yc)/(1+scale[1]))<1)

@Kernel.decorator
def square(size=(21,21),scale=(1,1), mode="fill"): 
    return _shapes_generator(size,scale, mode=mode, condition = lambda Xc,Yc, scale : ((Yc/scale[0]<=1) & (-1<Yc/scale[0]) & (-1<Xc/scale[1]) & (Xc/scale[1]<=1)))

@Kernel.decorator
def triangle(size=(21,21),scale=(1,1)): 
    return _shapes_generator(size,scale, condition = lambda Xc,Yc, scale : ((Yc/scale[0] <= 2) & (0>Xc/scale[1] -(Yc/scale[0]+1)) & (0 <  Yc/scale[0]+Xc/scale[1]+1)))

@Kernel.decorator
def cross(size=(21,21),scale=(1,1)): 
    return _shapes_generator(size,scale, condition = lambda Xc,Yc, scale : (Yc==0) & (Xc==0) & (Yc/scale[1]<=1)&(Yc/scale[1]>-1) & (Xc/scale[0]<=1)&(Xc/scale[1]>-1))