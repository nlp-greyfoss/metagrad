import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from datetime import datetime
import metagrad.module as nn
from metagrad.optim import Optimizer
from metagrad.tensor import Tensor


def use_svg_display():
    '''使用svg格式在jupyter中显示绘图'''
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    '''设置matplotlib的图标大小'''
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, title=None, saved_fname=None, random_fname=False,
         legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """通用画图类，修改自d2l包"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果 `X` 有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.title(title)
    plt.subplots_adjust(bottom=0.20)

    if random_fname:
        saved_fname = datetime.now().strftime("%Y%m%d%H%M%S%f")
    if saved_fname:
        plt.gcf().savefig(f"{saved_fname}.png", dpi=100)

    plt.show()


def to_onehot(y, num_classes=None):
    '''
    将标签值转换为one-hot向量
    :param y: 标签值 [0,1,2,...]
    :param num_classes: 类别数
    :return:
    '''
    if not num_classes:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]


def make_batches(X, y, batch_size=32, shuffle=True):
    '''
    将数据集拆分成批大小为batch_size的批数据
    :param X: 数据集 [样本数，样本维度]
    :param y: 对应的标签
    :param batch_size:  批大小
    :param shuffle: 是否需要对数据进行洗牌
    :return:
    '''
    n = X.shape[0]  # 样本数
    if shuffle:
        indexes = np.random.permutation(n)
    else:
        indexes = np.arange(n)

    X_batches = [
        Tensor(X[indexes, :][k:k + batch_size, :]) for k in range(0, n, batch_size)
    ]
    y_batches = [
        Tensor(y[indexes][k:k + batch_size]) for k in range(0, n, batch_size)
    ]

    return X_batches, y_batches


def loss_batch(model: nn.Module, loss_func, X_batch, y_batch, opt: Optimizer = None):
    '''
    对批数据计算损失
    :param model: 模型
    :param loss_func: 损失函数
    :param X_batch:  数据批次
    :param y_batch:  标签批次
    :param opt: 优化类
    :return: 损失值， 该批次大小
    '''
    loss = loss_func(model(X_batch), y_batch)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(X_batch)


def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == y_true)
