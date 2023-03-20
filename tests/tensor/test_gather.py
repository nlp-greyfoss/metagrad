import numpy as np

from metagrad.tensor import Tensor
import torch


def test_simple_gather():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = x.gather(1, np.array([[0, 0], [1, 0]]))

    t = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    ty = torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))

    assert y.tolist() == [[1, 1], [4, 3]]

    ty.sum().backward()
    y.sum().backward()

    assert np.allclose(x.grad, t.grad)


def test_by_column():
    data = np.arange(1, 17).reshape(4, -1).astype(np.float32)

    index = np.array([[0, 1, 2, 3],
                      [3, 2, 1, 0],
                      [2, 3, 0, 1],
                      [1, 2, 1, 0]])

    x = Tensor(data, requires_grad=True)
    # 按列选择
    y = x.gather(1, index)

    t = torch.tensor(data, requires_grad=True)
    ty = torch.gather(t, 1, torch.tensor(index, dtype=torch.int64))

    np.testing.assert_array_equal(y.data, ty.data)

    ty.sum().backward()
    y.sum().backward()

    assert np.allclose(x.grad, t.grad)


def test_by_row():
    data = np.arange(1, 17).reshape(4, -1).astype(np.float32)

    index = np.array([[0, 1, 2, 3],
                      [3, 2, 1, 0],
                      [2, 3, 0, 1],
                      [1, 2, 1, 0]])

    x = Tensor(data, requires_grad=True)
    # 按行选择
    y = x.gather(0, index)

    t = torch.tensor(data, requires_grad=True)
    ty = torch.gather(t, 0, torch.tensor(index, dtype=torch.int64))

    np.testing.assert_array_equal(y.data, ty.data)

    ty.sum().backward()
    y.sum().backward()

    assert np.allclose(x.grad, t.grad)
