import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_simple_cat():
    x = np.random.randn(2, 3)

    mx = Tensor(x, requires_grad=True)

    tx = torch.tensor(x, requires_grad=True)

    my = F.cat((mx, mx, mx), 0)
    ty = torch.cat((tx, tx, tx), 0)

    assert np.array_equal(my.data, ty.data)

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)
