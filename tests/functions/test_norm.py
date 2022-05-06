import numpy as np

from metagrad import Tensor
import metagrad.functions as F
import torch


def test_simple_norm_1():
    x = np.arange(9, dtype=np.float32) - 4
    x = x.reshape((3, 3))

    mx = Tensor(x, requires_grad=True)
    y = F.norm(mx, 1)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.norm(tx, p=1)

    assert np.allclose(y.data, ty.data)

    y.backward()
    ty.backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_simple_norm_2():
    x = np.arange(9, dtype=np.float32) - 4
    x = x.reshape((3, 3))

    mx = Tensor(x, requires_grad=True)
    y = F.norm(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.norm(tx)

    assert np.allclose(y.data, ty.data)

    y.backward()
    ty.backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_norm_1():
    x = np.array([[1, 2, 3], [-1, 1, 4]], dtype=np.float32)

    mx = Tensor(x, requires_grad=True)
    y = F.norm(mx, p=1, axis=0)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.norm(tx, p=1, dim=0)

    assert np.allclose(y.data, ty.data)

    y.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)


def test_norm_2():
    x = np.array([[1, 2, 3], [-1, 1, 4]], dtype=np.float32)

    mx = Tensor(x, requires_grad=True)
    y = F.norm(mx, axis=0)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.norm(tx, dim=0)

    assert np.allclose(y.data, ty.data)

    y.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)
