import numpy as np
from sklearn.model_selection import train_test_split

from metagrad.tensor import Tensor

import metagrad.module as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from metagrad.loss import CrossEntropyLoss
from metagrad.optim import SGD


class Feedforward(nn.Module):
    '''
    简单单隐藏层前馈网络，用于分类问题
    '''

    def __init__(self, input_size, hidden_size, output_size):
        '''

        :param input_size: 输入维度
        :param hidden_size: 隐藏层大小
        :param output_size: 分类个数
        '''
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 隐藏层，将输入转换为隐藏向量
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_size, output_size)  # 输出层，将隐藏向量转换为输出
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# 生成数据
def generate_dataset(draw_picture=False):
    # 生成具有4个类别、2个特征(方便画图)的1000个样本点
    X, y = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=666)

    if draw_picture:
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.legend()
        fig = plt.gcf()
        fig.savefig('blobs.png', dpi=100)
        plt.show()

    y = np.eye(4)[y] # 转换成one-hot

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    return Tensor(X_train), Tensor(X_test), Tensor(y_train), Tensor(y_test)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = generate_dataset()
    epochs = 2000

    model = Feedforward(2, 4, 4)  # 输入2D,隐藏层4D，输出类别4个

    optimizer = SGD(model.parameters(), lr=0.1)
    # 负对数似然
    loss = CrossEntropyLoss()

    losses = []

    for epoch in range(int(epochs)):
        outputs = model(X_train)
        l = loss(outputs, y_train)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            losses.append(l.item())
            print(f"Train -  Loss: {l.item()}")

    # 在测试集上测试
    outputs = model(X_test)
    correct = np.sum(outputs.numpy().argmax(-1) == y_test.numpy().argmax(-1))
    accuracy = 100 * correct / len(y_test)
    print(f"Test Accuracy:{accuracy}")
