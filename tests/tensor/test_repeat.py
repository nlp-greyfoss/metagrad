import numpy as np

from metagrad.tensor import Tensor
import torch


def test_simple_repeat():
    mx = Tensor([1., 2., 3.], requires_grad=True)
    mz = mx.repeat(2)

    tx = torch.tensor([1., 2., 3.], requires_grad=True)
    tz = tx.repeat(2)

    np.allclose(mz.array(), tz.data)

    tz.sum().backward()

    mz.sum().backward()

    assert np.allclose(mx.grad.data,
                       tx.grad.data)


def test_repeat():
    mx = Tensor([1., 2., 3.], requires_grad=True)
    mz = mx.repeat((4, 2))

    tx = torch.tensor([1., 2., 3.], requires_grad=True)
    tz = tx.repeat(4, 2)

    np.allclose(mz.array(), tz.data)

    tz.sum().backward()

    mz.sum().backward()

    assert np.allclose(mx.grad.data,
                       tx.grad.data)
