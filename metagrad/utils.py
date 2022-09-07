import os
import time
from datetime import datetime
from typing import List

import imageio
import numpy as np
from matplotlib import pyplot as plt

import metagrad.module as nn
from metagrad.dataloader import DataLoader
from metagrad.optim import Optimizer
from metagrad.tensor import Tensor
import metagrad.functions as F


def set_figsize(figsize=(4.9, 3.5)):
    '''设置matplotlib的图标大小'''
    plt.rcParams['figure.figsize'] = figsize
    plt.subplots_adjust(bottom=0.20)


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
    plt.tight_layout()

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
    return np.mean(np.argmax(y_pred.array(), axis=1) == y_true.array())


def regression_classification_metric(y_pred, y_true):
    '''
    回归问题当成分类时的评价指标
    :param y_pred:
    :param y_true:
    :return:
    '''
    return np.sum(y_pred.array().round() == y_true.array()) / len(y_pred)


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
                 figsize=(4.9, 3.5), saved_file='animator', plot_show=True):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.saved_file = saved_file
        self.filenames = []
        self.file_index = 0
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def _save_image(self, image_name=None):
        if image_name is None:
            image_name = self.saved_file
        plt.tight_layout()
        plt.savefig(image_name)

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
        # 生成中间文件
        filename = f'{self.file_index}.png'

        self._save_image(filename)

        self.filenames.append(filename)
        self.file_index = self.file_index + 1

    def show(self):
        with imageio.get_writer(f'{self.saved_file}.gif', mode='I') as writer:
            for filename in self.filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in self.filenames:
            os.remove(filename)

        self.filenames = []
        self.file_index = 0

        self._save_image(f'{self.saved_file}.png')

        plt.show()
        plt.close()


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def run_epoch(model: nn.Module, data_loader: DataLoader, loss: nn.Module, opt: Optimizer = None,
              activate_func=lambda x: x, evaluate_func=accuracy):
    '''
    进行一次迭代
    :param data_loader: 数据加载器
    :param model: 模型
    :param loss: 损失函数，reduction=None
    :param opt: 优化器
    :param activate_func: 网络最后一层补上的激活函数，默认原样输出
    :param evaluate_func: 评价函数，默认为准确率

    :return: 损失 和 准确率
    '''
    assert loss.reduction is None, "loss.reduction must be null."

    metric = Accumulator(3)

    for X_batch, y_batch in data_loader:
        y_pred = model(X_batch)
        l = loss(y_pred, y_batch)

        if opt is not None:
            l.mean().backward()  # 相当于计算了均方误差
            opt.step()
            opt.zero_grad()

        metric.add(l.sum().item(), evaluate_func(activate_func(y_pred), y_batch), y_batch.size())
    # 总损失 / 样本总数 ， 总准确率 / 样本总数
    return metric[0] / metric[2], metric[1] / metric[2]


def pad_sequence(sequences: List[Tensor], padding_value: int = 0) -> Tensor:
    max_len = max(len(x) for x in sequences)

    shape = (len(sequences), max_len) + sequences[0].shape[1:]
    y = Tensor.empty(shape, dtype=sequences[0].dtype)
    for i, x in enumerate(sequences):
        l = len(x)
        if l == max_len:
            y[i] = x
        else:
            y[i, 0:l] = x
            y[i, l:] = padding_value

    return y


def unpad_sequence(padded_sequences, lengths):
    unpadded_sequences = []

    max_len = padded_sequences.shape[1]
    idx = Tensor.arange(max_len)

    for seq, length in zip(padded_sequences, lengths):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    if isinstance(parameters, Tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return Tensor(0.)

    device = grads[0].device

    if norm_type == float('inf'):
        norms = [g.abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else F.stack(norms).max()
    else:
        total_norm = F.norm(F.stack([F.norm(g, norm_type).to(device) for g in grads]), norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = F.clamp(clip_coef, max=1.0)
    for g in grads:
        g.mul_(clip_coef_clamped.to(g.device))
    return total_norm
