
import kerpy, numpy as np
from kerpy.objs.Kernel import Kernel

def test_laplacian_finite_relation():
    assert np.sum(np.sum(list((kerpy.diff.finite() + kerpy.diff.finite(flip=False)).to_reals().values()), axis=0) == kerpy.diff.laplacian().numpy) == 9