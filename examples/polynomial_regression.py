import math

import numpy as np

from metagrad.loss import MSELoss
from metagrad.optim import SGD
from metagrad.tensor import Tensor
import metagrad.module as nn
from metagrad.utils import Animator

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
# 真实w只有前4位有效
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
# 多项式特征
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))

for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!

# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
# 增加均值为0标准差为0.1正态分布的噪声项
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为Tensor
true_w, features, poly_features, labels = [Tensor(x) for x in [true_w, features, poly_features, labels]]

print(features[:2], poly_features[:2, :], labels[:2])


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = MSELoss()
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    model = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)),
                               batch_size, is_train=False)
    trainer = SGD(model.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch(model, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
