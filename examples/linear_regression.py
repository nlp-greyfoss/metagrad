import numpy as np

from metagrad.loss import MSELoss
from metagrad.module import Linear
from metagrad.optim import SGD
from metagrad.tensor import Tensor



def simple_demo():
    model = Linear(1, 1)

    optimizer = SGD(model.parameters(), lr=1e-4)

    loss = MSELoss()

    # 面积
    areas = [64.4, 68, 74.1, 74., 76.9, 78.1, 78.6]
    # 房龄
    ages = [31, 21, 19, 24, 17, 16, 17]

    X = np.vstack([areas, ages])
    # 挂牌售价
    prices = [6.1, 6.25, 7.8, 6.66, 7.82, 7.14, 8.02]

    X = Tensor(areas).reshape((-1, 1))
    y = Tensor(prices).reshape((-1, 1))

    for epoch in range(500):
        l = loss(model(X), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        epoch_loss = l.data
        print(f'epoch {epoch + 1}, loss {float(epoch_loss):f}')

    print(model.weight)
    print(model.bias)


if __name__ == '__main__':
    simple_demo()
