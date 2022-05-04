from metagrad.functions import *
from metagrad.utils import plot


def plot_relu():
    x = Tensor.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = relu(x)
    plot(x.array(), y.array(), 'x', 'relu(x)', random_fname=True, figsize=(5, 2.5))

    y.backward(Tensor.ones_like(x))
    plot(x.array(), x.grad.array(), 'x', 'grad of relu', random_fname=True, figsize=(5, 2.5))


def plot_leaky_relu():
    x = Tensor.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = leaky_relu(x, slope=0.1)
    plot(x.array(), y.array(), 'x', 'leaky relu(x)', random_fname=True, figsize=(5, 2.5))

    y.backward(Tensor.ones_like(x))
    plot(x.array(), x.grad.array(), 'x', 'grad of leaky relu', random_fname=True, figsize=(5, 2.5))


def plot_elu():
    x = Tensor.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = elu(x)
    plot(x.array(), y.array(), 'x', 'elu(x)', random_fname=True, figsize=(5, 2.5))

    y.backward(Tensor.ones_like(x))
    plot(x.array(), x.grad.array(), 'x', 'grad of elu', random_fname=True, figsize=(5, 2.5))


def plot_swish():
    x = Tensor.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = swish(x)
    plot(x.array(), y.array(), 'x', 'swish(x)', random_fname=True, figsize=(5, 2.5))

    y.backward(Tensor.ones_like(x))
    plot(x.array(), x.grad.array(), 'x', 'grad of swish', random_fname=True, figsize=(5, 2.5))


def plot_softplus():
    x = Tensor.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = softplus(x)
    plot(x.array(), y.array(), 'x', 'softplus(x)', random_fname=True, figsize=(5, 2.5))

    y.backward(Tensor.ones_like(x))
    plot(x.array(), x.grad.array(), 'x', 'grad of softplus', random_fname=True, figsize=(5, 2.5))


def plot_sigmoid():
    x = Tensor.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = sigmoid(x)
    plot(x.array(), y.array(), 'x', 'sigmoid(x)', random_fname=True, figsize=(5, 2.5))

    y.backward(Tensor.ones_like(x))
    plot(x.array(), x.grad.array(), 'x', 'grad of sigmoid', random_fname=True, figsize=(5, 2.5))


def plot_tanh():
    x = Tensor.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = tanh(x)
    plot(x.array(), y.array(), 'x', 'tanh(x)', random_fname=True, figsize=(5, 2.5))

    y.backward(Tensor.ones_like(x))
    plot(x.array(), x.grad.array(), 'x', 'grad of tanh', random_fname=True, figsize=(5, 2.5))


if __name__ == '__main__':
    plot_tanh()
