import numpy as np

from metagrad.module import Linear
from metagrad.optim import SGD
import metagrad.functions as F
from metagrad.tensor import Tensor


def test_weight_decay():
    weight_decay = 0.5
    X = Tensor(np.random.rand(5, 2))
    y = np.array([0, 1, 2, 2, 0]).astype(np.int32)
    y = Tensor(np.eye(3)[y])
    model = Linear(2, 3, False)
    model.weight.assign(np.ones_like(model.weight.data))
    # 带有weigth_decay
    optimizer = SGD(params=model.parameters(), weight_decay=weight_decay)

    pred = model(X)
    loss = F.cross_entropy(pred, y)
    loss.backward()
    optimizer.step()
    weight_0 = model.weight.data.copy()

    # 重新设置权重
    model.weight.assign(np.ones_like(model.weight.data))
    # 没有weigth_decay
    optimizer = SGD(params=model.parameters())
    model.zero_grad()

    pred = model(X)
    # 在原来的loss上加上L2正则
    loss = F.cross_entropy(pred, y) + weight_decay / 2 * (model.weight ** 2).sum()
    loss.backward()

    optimizer.step()
    weight_1 = model.weight.data.copy()
    print(weight_0.data)
    print(weight_1.data)

    assert np.allclose(weight_0.data, weight_1.data)
