import numpy as np

from metagrad.tensor import Tensor


def test_simple_index_fill():
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    index = Tensor([0, 2])
    x.index_fill_(1, index, -1)

    assert x.data.tolist() == [[-1., 2., - 1.],
                               [-1., 5., - 1.],
                               [-1., 8., - 1.]]


def test_index_fill():
    x = Tensor.ones(10)
    # [1 3 5 7]
    index = Tensor(np.arange(1, 9, 2))
    # 将 1 3 5 7处的值置为-1
    x.index_fill_(0, index, -1)
    assert x.data.tolist() == [1., -1., 1., -1., 1., -1., 1., -1., 1., 1.]
