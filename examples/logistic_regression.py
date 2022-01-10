import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import metagrad.functions as F
from metagrad.loss import BCELoss
from metagrad.module import Module, Linear
from metagrad.optim import SGD
from metagrad.tensor import Tensor
from tqdm import tqdm


class LogisticRegression(Module):
    def __init__(self, input_dim, output_dim):
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(self.linear(x))


def model_plot(model, X, y, title):
    import matplotlib.pyplot as plt

    w = model.linear.weight
    b = model.linear.bias

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet')

    u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
    plt.plot(u, (0.5 - b - w[0] * u) / w[1])
    plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    plt.xlabel(r'$\boldsymbol{x_1}$', fontsize=16)
    plt.ylabel(r'$\boldsymbol{x_2}$', fontsize=16)
    plt.title(title)
    plt.show()


def load_dataset():
    samples = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1,
                                  n_clusters_per_class=1, flip_y=-1)
    red = samples[0][samples[1] == 0]
    blue = samples[0][samples[1] == 1]
    red_labels = np.zeros(len(red))
    blue_labels = np.ones(len(blue))

    labels = np.append(red_labels, blue_labels)
    inputs = np.concatenate((red, blue), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.33, random_state=42)

    return Tensor(X_train), Tensor(X_test), Tensor(y_train.reshape(-1, 1)), Tensor(y_test.reshape(-1, 1))


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_dataset()

    epochs = 200000

    model = LogisticRegression(2, 1)

    optimizer = SGD(model.parameters(), lr=1e-3)

    loss = BCELoss()

    losses = []

    for epoch in tqdm(range(int(epochs))):

        x = X_train
        labels = y_train
        optimizer.zero_grad()
        outputs = model(X_train)
        l = loss(outputs, labels)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if (epoch + 1) % 10000 == 0:
            total = 0
            correct = 0
            total += len(y_train)
            correct += np.sum(outputs.numpy().round() == y_train.numpy())
            accuracy = 100 * correct / total
            losses.append(l.item())

            print(f"Train -  Loss: {l.item()}. Accuracy: {accuracy}\n")
