import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from metagrad.functions import sigmoid
from metagrad.loss import BCELoss
from metagrad.module import Module, Linear
from metagrad.optim import SGD
from metagrad.tensor import Tensor


class LogisticRegression(Module):
    def __init__(self, input_dim, output_dim):
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def load_data(path, draw_picture=False):
    data = pd.read_csv(path)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if draw_picture:
        # filter out the applicants that got admitted
        admitted = data.loc[y == 1]

        # filter out the applicants that din't get admission
        not_admitted = data.loc[y == 0]

        # plots
        plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
        plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
        plt.xlabel('Marks in 1st Exam')
        plt.ylabel('Marks in 2nd Exam')
        plt.legend()
        fig = plt.gcf()
        fig.savefig('marks.png', dpi=100)

    y = y[:, np.newaxis]

    return Tensor(X), Tensor(y)


if __name__ == '__main__':

    X, y = load_data("./data/marks.txt", draw_picture=True)

    epochs = 200000

    model = LogisticRegression(2, 1)

    # model.linear.weight.assign([[-0.29942604, 0.78735491]]) 有问题的权重
    print(f"using weight: {model.linear.weight}")
    optimizer = SGD(model.parameters(), lr=1e-3)

    loss = BCELoss()

    losses = []

    for epoch in tqdm(range(int(epochs))):
        outputs = model(X)
        l = loss(outputs, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if (epoch + 1) % 10000 == 0:
            total = 0
            correct = 0
            total += len(y)
            correct += np.sum(sigmoid(outputs).numpy().round() == y.numpy())
            accuracy = 100 * correct / total
            losses.append(l.item())

            print(f"Train -  Loss: {l.item()}. Accuracy: {accuracy}\n")

    weight = model.linear.weight.squeeze()
    bias = model.linear.bias.squeeze()

    print(weight, bias)

    X_numpy = X.numpy()
    x_values = [np.min(X_numpy[:, 0] - 2), np.max(X_numpy[:, 1] + 2)]
    y_values = - (bias + np.dot(weight[0], x_values)) / weight[1]

    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Marks in 1st Exam')
    plt.ylabel('Marks in 2nd Exam')
    plt.legend()
    fig = plt.gcf()
    fig.savefig('marks_decision.png')
