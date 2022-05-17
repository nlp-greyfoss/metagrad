import torch

from metagrad.tensor import Tensor
import numpy as np
import metagrad.functions as F


def test_simple_squeeze():
    x = np.zeros((2, 1, 2, 1, 2))

    mx = Tensor(x, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    my = F.squeeze(mx)
    ty = torch.squeeze(tx)

    assert my.shape == ty.shape

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_squeeze():
    x = np.zeros((2, 1, 2, 1, 2))

    mx = Tensor(x, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    my = F.squeeze(mx, 1)
    ty = torch.squeeze(tx, 1)

    assert my.shape == ty.shape

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_complex_squeeze():
    x = np.zeros((1, 3, 1, 1, 2))

    x = Tensor(x, requires_grad=True)

    y = F.squeeze(x, (0, 2, -2))

    assert y.shape == (3, 2)

    y.sum().backward()

    assert x.grad.shape == (1, 3, 1, 1, 2)
