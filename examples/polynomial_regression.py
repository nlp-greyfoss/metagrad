import math

import numpy as np

import metagrad.module as nn
from metagrad.loss import MSELoss
from metagrad.optim import SGD
from metagrad.tensor import no_grad
from metagrad.utils import Animator, run_epoch

max_degree = 30  # 多项式的最大阶数
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
# true_w, features, poly_features, labels = [Tensor(x) for x in [true_w, features, poly_features, labels]]

print(f"{features[:2]}\n {poly_features[:2, :]}\n  {labels[:2]}\n")


def train(train_features, test_features, train_labels, test_labels, num_epochs=400, fname="animator"):
    loss = MSELoss(reduction=None)
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    model = nn.Linear(input_shape, 1, bias=False)
    batch_size = min(10, train_labels.shape[0])

    opt = SGD(model.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                        legend=['train', 'test'], saved_file=fname)

    train_labels, test_labels = train_labels.reshape(-1, 1), test_labels.reshape(-1, 1)

    for epoch in range(num_epochs):
        train_loss, _ = run_epoch(model, train_features, train_labels, loss, opt, batch_size)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            with no_grad():
                test_loss, _ = run_epoch(model, test_features, test_labels, loss, opt=None,
                                         batch_size=batch_size)
            animator.add(epoch + 1, (train_loss, test_loss))

    animator.show()
    print('weight:', model.weight.numpy())


# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:],
      fname="good_fitting")

# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:],
      fname="under_fitting")

train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500,
      fname="over_fitting")
