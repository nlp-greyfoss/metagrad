from metagrad.tensor import Tensor
import numpy as np


def test_masked_fill():
    x = Tensor.randn(5, 6, requires_grad=True)
    seq_len = [5, 4, 3, 2, 1]
    mask = Tensor.zeros((5, 6))
    for e_id, src_len in enumerate(seq_len):
        mask[e_id, src_len:] = 1

    y = x.masked_fill(mask.bool(), 2.0)

    y.sum().backward()

    assert x.grad.tolist() == [[1., 1., 1., 1., 1., 0.],
                               [1., 1., 1., 1., 0., 0.],
                               [1., 1., 1., 0., 0., 0.],
                               [1., 1., 0., 0., 0., 0.],
                               [1., 0., 0., 0., 0., 0.]]
