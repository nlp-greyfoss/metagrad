import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from tqdm import tqdm

from metagrad.loss import CrossEntropyLoss

from metagrad.module import Module, Linear
from metagrad.optim import SGD
from metagrad.tensor import Tensor


class SoftmaxRegression(Module):
    def __init__(self, input_dim, output_dim):
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # 只要输出logits即可
        return self.linear(x)


def generate_dataset(draw_picture=False):
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 我们只需要前两个特征
    y = iris.target

    if draw_picture:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        # Plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor="k")
        plt.xlabel("Sepal length")
        plt.ylabel("Sepal width")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())

        fig = plt.gcf()
        fig.savefig('iris.png', dpi=100)

    y = y[:, np.newaxis]
    return Tensor(X), Tensor(y)


if __name__ == '__main__':
    X, y = generate_dataset(True)

    epochs = 200_000

    model = SoftmaxRegression(2, 3)  # 2个特征 3个输出

    optimizer = SGD(model.parameters(), lr=1e-3)

    loss = CrossEntropyLoss()

    losses = []

    for epoch in tqdm(range(int(epochs))):

        optimizer.zero_grad()
        outputs = model(X)
        l = loss(outputs, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if (epoch + 1) % 10000 == 0:
            losses.append(l.item())
            print(f"Train -  Loss: {l.item()}")
