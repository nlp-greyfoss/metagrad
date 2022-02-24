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


class Accumulator:
    '''
    在n个变量上累加
    比如可用于遍历批数据，累加判断正确样本数以及遍历的样本总数
    '''

    def __init__(self, n: int):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据，改自d2l"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_batch(model: nn.Module, X_batch, y_batch, loss: nn.Module, opt: Optimizer):
    # 训练损失总和、训练准确度总和、样本总数
    metric = Accumulator(3)

    y_pred = model(X_batch)
    l = loss(y_pred, y_batch)

    l.backward()
    opt.step()
    opt.zero_grad()

    metric.add(l.sum())
