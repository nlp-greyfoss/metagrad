import numpy as np

from metagrad.functions import sigmoid
from metagrad.tensor import Tensor

import metagrad.module as nn
from keras.datasets import imdb
from metagrad.loss import BCELoss
from metagrad.optim import SGD
from metagrad.utils import make_batches, loss_batch, accuracy
from metagrad.tensor import no_grad

import matplotlib.pyplot as plt


class Feedforward(nn.Module):
    '''
    简单单隐藏层前馈网络，用于分类问题
    '''

    def __init__(self, input_size, hidden_size, output_size):
        '''

        :param input_size: 输入维度
        :param hidden_size: 隐藏层大小
        :param output_size: 分类个数
        '''
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 隐藏层，将输入转换为隐藏向量
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_size, output_size)  # 输出层，将隐藏向量转换为输出
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# 加载数据集
def load_dataset():
    # 保留训练数据中前10000个最常出现的单词，舍弃低频单词
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
    # 标签的维度很重要，否则训练不起来
    y_train, y_test = y_train[:, np.newaxis], y_test[:, np.newaxis]
    return X_train, X_test, y_train, y_test


def indices_to_sentence(indices: Tensor):
    # 单词索引字典 word -> index
    word_index = imdb.get_word_index()
    # 逆单词索引字典 index -> word
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    # 将index列表转换为word列表
    #
    # 0、1、2 是为“padding”（填充）、“start of sequence”（序
    # 列开始）、“unknown”（未知词）分别保留的索引
    decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in indices.data])
    return decoded_review


def vectorize_sequences(sequences, dimension=10000):
    # 默认生成一个[句子长度，维度数]的向量
    results = np.zeros((len(sequences), dimension), dtype='uint8')
    for i, sequence in enumerate(sequences):
        # 将第i个序列中，对应单词序号处的位置置为1
        results[i, sequence] = 1
    return results


def compute_loss_and_accury(X_batches, y_batches, model, loss_func, total_nums, opt=None):
    losses = []
    correct = 0
    for X_batch, y_batch in zip(X_batches, y_batches):
        y_pred = model(X_batch)
        l = loss_func(y_pred, y_batch)

        if opt is not None:
            l.backward()
            opt.step()
            opt.zero_grad()

        # 当前批次的损失
        losses.append(l.item())

        correct += np.sum(sigmoid(y_pred).numpy().round() == y_batch.numpy())

    loss = sum(losses) / total_nums  # 总损失 除以 样本总数
    accuracy = 100 * correct / total_nums

    return loss, accuracy


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset()

    X_train = vectorize_sequences(X_train)
    X_test = vectorize_sequences(X_test)

    # 保留验证集
    # X_train有25000条数据，我们保留10000条作为验证集
    X_val = X_train[:10000]
    X_train = X_train[10000:]

    y_val = y_train[:10000]
    y_train = y_train[10000:]

    model = Feedforward(10000, 128, 1)  # 输入大小10000,隐藏层大小128，输出只有一个，代表判断为正例的概率

    optimizer = SGD(model.parameters(), lr=0.001)
    # 先计算sum
    loss = BCELoss(reduction="sum")

    epochs = 20
    batch_size = 512
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # 由于数据过多，需要拆分成批次
    X_train_batches, y_train_batches = make_batches(X_train, y_train, batch_size=batch_size)

    X_val_batches, y_val_batches = make_batches(X_val, y_val, batch_size=batch_size)

    for epoch in range(epochs):
        # losses, nums = zip(*[loss_batch(model, loss, X_batch, y_batch, optimizer)
        #                     for X_batch, y_batch in zip(X_train_batches, y_train_batches)])

        train_loss, train_accuracy = compute_loss_and_accury(X_train_batches, y_train_batches, model, loss,
                                                             len(X_train), optimizer)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        with no_grad():
            val_loss, val_accuracy = compute_loss_and_accury(X_val_batches, y_val_batches, model, loss, len(X_val))

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        print(f"Epoch:{epoch + 1}, Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}% | "
              f" Validation Loss:{val_loss:.4f} , Accuracy:{val_accuracy:.2f}%")

    # 绘制训练损失和验证损失
    epoch_list = range(1, epochs + 1)
    plt.plot(epoch_list, train_losses, 'r', label='Training loss')
    plt.plot(epoch_list, val_losses, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制训练准确率和验证准确率
    # 清空图像
    plt.clf()
    plt.plot(epoch_list, train_accuracies, 'r', label='Training acc')
    plt.plot(epoch_list, val_accuracies, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 最后在测试集上测试
    with no_grad():
        X_test, y_test = Tensor(X_test), Tensor(y_test)
        outputs = model(X_test)
        correct = np.sum(sigmoid(outputs).numpy().round() == y_test.numpy())
        accuracy = 100 * correct / len(y_test)
        print(f"Test Accuracy:{accuracy}")