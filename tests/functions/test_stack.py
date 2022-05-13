import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_simple_stack():
    x1 = np.array([1., 3., 5., 7.])
    x2 = np.array([2., 4., 6., 7.])

    mx1 = Tensor(x1, requires_grad=True)
    mx2 = Tensor(x2)

    tx1 = torch.tensor(x1, requires_grad=True)
    tx2 = torch.tensor(x2)

    my = F.stack((mx1, mx2))
    ty = torch.stack((tx1, tx2))

    assert np.allclose(my.data, ty.data)

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx1.grad.data, tx1.grad.data)
