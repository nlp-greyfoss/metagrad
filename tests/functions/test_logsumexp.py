import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_simple_logsumexp():
    x = np.array([0, 1, 2], np.float32)

    mx = Tensor(x, requires_grad=True)
    y = F.logsumexp(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.logsumexp(tx, -1)

    assert np.allclose(y.data, ty.data)

    y.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data, rtol=1e-05, atol=1e-05)


def test_logsoftmax():
    N, CLS_NUM = 100, 10  # 样本数，类别数
    x = np.random.randn(N, CLS_NUM)

    x = np.array(x, np.float32)

    mx = Tensor(x, requires_grad=True)
    y = F.logsumexp(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.logsumexp(tx, dim=1, keepdim=True)

    assert np.allclose(y.data, ty.data)

    y.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad.data, tx.grad.data)
