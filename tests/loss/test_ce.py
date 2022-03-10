import numpy as np
import torch
from metagrad.loss import CrossEntropyLoss
from metagrad.tensor import Tensor

def test_ce_loss():
    N, CLS_NUM = 100, 10  # 样本数，类别数
    x = np.random.randn(N, CLS_NUM)
    t = np.random.randint(0, CLS_NUM, (N,))

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(np.eye(x.shape[-1])[t])  # 需要转换成one-hot向量

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.int64)

    my_loss = CrossEntropyLoss()
    torch_loss = torch.nn.CrossEntropyLoss()

    ml = my_loss(mx, mt)
    tl = torch_loss(tx, tt)

    assert np.allclose(ml.item(), tl.item())

    ml.backward()
    tl.backward()

    assert np.allclose(mx.grad.data, tx.grad.data)
