import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_simple_flip():
    x = np.random.randn(2, 2, 2)

    mx = Tensor(x, requires_grad=True)

    tx = torch.tensor(x, requires_grad=True)

    my = F.flip(mx, 0)
    ty = torch.flip(tx, (0,))

    assert np.array_equal(my.data, ty.data)

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad, tx.grad)


def test_flip():
    x = np.random.randn(2, 2, 2)

    mx = Tensor(x, requires_grad=True)

    tx = torch.tensor(x, requires_grad=True)

    my = F.flip(mx, (0, 1))
    ty = torch.flip(tx, (0, 1))

    assert np.array_equal(my.data, ty.data)

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad, tx.grad)
