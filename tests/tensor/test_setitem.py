import metagrad.functions as F
from metagrad.tensor import Tensor, debug_mode, cuda
import numpy as np


def test_setitem():
    # x = Tensor([1, 2, 3, 4, 5, 6, 7], requires_grad=True)
    # x[2:4] = 1
    #
    # assert x.tolist() == [1, 2, 1, 1, 5, 6, 7]
    # x.backward(np.array([1] * 7))
    #
    # assert x.grad.tolist() == [0, 0, 1, 1, 0, 0, 0]
    pass

