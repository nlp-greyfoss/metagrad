import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from metagrad.loss import NLLLoss
from metagrad.module import Module, Linear
from metagrad.optim import SGD
from metagrad.tensor import Tensor
import metagrad.functions as F


class SoftmaxRegression(Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # 计算对数概率
        return F.log_softmax(self.linear(x))


def generate_dataset(draw_picture=False):
    iris = datasets.load_iris()

    X = iris['data']
    y = iris['target']
    names = iris['target_names']  # 类名
    feature_names = iris['feature_names']  # 特征名

    if draw_picture:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        for target, target_name in enumerate(names):
            X_plot = X[y == target]
            plt.plot(X_plot[:, 0], X_plot[:, 1],
                     linestyle='none',
                     marker='o',
                     label=target_name)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.axis('equal')
        plt.legend()

        fig = plt.gcf()
        fig.savefig('iris.png', dpi=100)

    y = np.eye(3)[y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2)

    return Tensor(X_train), Tensor(X_test), Tensor(y_train), Tensor(y_test)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = generate_dataset(True)
    epochs = 2000

    model = SoftmaxRegression(4, 3)  # 4个特征 3个输出

    optimizer = SGD(model.parameters(), lr=1e-1)
    # 负对数似然
    loss = NLLLoss()

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
    correct = np.sum(outputs.array().argmax(-1) == y_test.array().argmax(-1))
    accuracy = 100 * correct / len(y_test)
    print(f"Test Accuracy:{accuracy}")
