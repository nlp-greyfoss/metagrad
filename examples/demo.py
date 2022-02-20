from metagrad.tensor import Tensor
from metagrad.functions import tanh
from metagrad.utils import plot

if __name__ == '__main__':
    x = Tensor.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = tanh(x)
    plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', random_fname=True, figsize=(5, 2.5))
