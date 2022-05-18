import torch

from metagrad.tensor import Tensor
import numpy as np


def test_simple_squeeze():
    x = np.zeros((2, 1, 2, 1, 2))

    mx = Tensor(x, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    my = mx.squeeze()
    ty = tx.squeeze()

    assert my.shape == ty.shape

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_squeeze():
    x = np.zeros((2, 1, 2, 1, 2))

    mx = Tensor(x, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    my = mx.squeeze(1)
    ty = tx.squeeze(1)

    assert my.shape == ty.shape

    my.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_complex_squeeze():
    x = np.zeros((1, 3, 1, 1, 2))

    x = Tensor(x, requires_grad=True)

    y = x.squeeze((0, 2, -2))

    assert y.shape == (3, 2)

    y.sum().backward()

    assert x.grad.shape == (1, 3, 1, 1, 2)
