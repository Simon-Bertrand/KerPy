
import kerpy, numpy as np
from kerpy.objs.Kernel import Kernel

def test_laplacian_sharpen_relation():
    l : Kernel = kerpy.processing.laplacian()
    l.numpy[1,1] += 1
    assert np.sum((kerpy.processing.sharpen()==l).numpy) == 9