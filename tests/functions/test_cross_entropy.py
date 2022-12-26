import numpy as np
import torch

import metagrad.functions as F
from metagrad.tensor import Tensor


def test_binary_cross_entropy():
    N = 10
    x = np.random.randn(N)
    y = np.random.randint(0, 1, (N,))

    mx = Tensor(x, requires_grad=True)
    my = Tensor(y)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    ty = torch.tensor(y, dtype=torch.float32)

    mo = torch.binary_cross_entropy_with_logits(tx, ty).mean()
    to = F.binary_cross_entropy(mx, my)

    assert np.allclose(mo.data,
                       to.array())

    mo.backward()
    to.backward()

    assert np.allclose(mx.grad,
                       tx.grad)


def test_cross_entropy():
    x = np.array([[0, 1, 2, 3], [4, 0, 2, 1]], np.float32)
    t = np.array([3, 0]).astype(np.int32)

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(np.eye(x.shape[-1])[t])  # 需要转换成one-hot向量

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.int64)

    to = torch.nn.functional.cross_entropy(tx, tt)
    mo = F.cross_entropy(mx, mt)
    assert mo.item() == to.item()

    mo.backward()
    to.backward()

    assert np.allclose(mx.grad, tx.grad)


def test_cross_entropy_class_indices():
    x = np.array([[0, 1, 2, 3], [4, 0, 2, 1]], np.float32)
    t = np.array([3, 0])

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(t)  # 不需要转换为one-hot向量

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.LongTensor(t)

    to = torch.nn.functional.cross_entropy(tx, tt)
    mo = F.cross_entropy(mx, mt)
    assert mo.item() == to.item()

    mo.backward()
    to.backward()

    assert np.allclose(mx.grad, tx.grad)


def test_random():
    N, CLS_NUM = 100, 10  # 样本数，类别数
    x = np.random.randn(N, CLS_NUM)
    t = np.random.randint(0, CLS_NUM, (N,))

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(np.eye(x.shape[-1])[t])  # 需要转换成one-hot向量

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.int64)

    to = torch.nn.functional.cross_entropy(tx, tt)
    mo = F.cross_entropy(mx, mt)

    assert np.allclose(mo.data, to.data)

    mo.backward()
    to.backward()

    assert np.allclose(mx.grad, tx.grad)
