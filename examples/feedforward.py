import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import imdb

import metagrad.module as nn
from metagrad.dataloader import DataLoader
from metagrad.dataset import TensorDataset
from metagrad.functions import sigmoid
from metagrad.loss import BCELoss
from metagrad.optim import SGD, Adam
from metagrad.tensor import Tensor, debug_mode
from metagrad.tensor import no_grad
from metagrad import cuda


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
        super(Feedforward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 隐藏层，将输入转换为隐藏向量
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_size, output_size)  # 输出层，将隐藏向量转换为输出
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# 加载数据集
def load_dataset():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
    # 标签的维度很重要，否则训练不起来
    y_train, y_test = y_train[:, np.newaxis].astype(np.uint8), y_test[:, np.newaxis].astype(np.uint8)
    X_train = vectorize_sequences(X_train)
    X_test = vectorize_sequences(X_test)

    # 保留验证集
    # X_train有25000条数据，我们保留10000条作为验证集
    X_val = X_train[:10000]
    X_train = X_train[10000:]

    y_val = y_train[:10000]
    y_train = y_train[10000:]

    return Tensor(X_train), Tensor(X_test), Tensor(y_train), Tensor(y_test), Tensor(X_val,), Tensor(y_val)


def indices_to_sentence(indices: Tensor):
    # 单词索引字典 word -> indices
    word_index = imdb.get_word_index()
    # 逆单词索引字典 indices -> word
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


def compute_loss_and_accury(data_loader: DataLoader, model, loss_func, total_nums, opt=None, device=None):
    losses = []
    correct = 0
    for X_batch, y_batch in data_loader:
        X_batch.to(device)
        y_batch.to(device)

        y_pred = model(X_batch)
        l = loss_func(y_pred, y_batch)
        if opt is not None:
            l.backward()
            opt.step()
            opt.zero_grad()

        # 当前批次的损失
        losses.append(l.item())

        correct += np.sum(sigmoid(y_pred).array().round() == y_batch.array())

    loss = sum(losses) / total_nums  # 总损失 除以 样本总数
    accuracy = 100 * correct / total_nums

    return loss, accuracy.item()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X_val, y_val = load_dataset()

    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
    print(f"current device: {device}")

    model = Feedforward(10000, 128, 1)  # 输入大小10000,隐藏层大小128，输出只有一个，代表判断为正例的概率
    model.to(device)

    optimizer = SGD(model.parameters(), lr=0.001)
    # 先计算sum
    loss = BCELoss(reduction="sum")

    epochs = 20
    batch_size = 512
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # 由于数据过多，需要拆分成批次，使用自定义数据集和加载器
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size)

    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(epochs):
        train_loss, train_accuracy = compute_loss_and_accury(train_dl, model, loss, len(X_train), optimizer,
                                                             device=device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        with no_grad():
            val_loss, val_accuracy = compute_loss_and_accury(val_dl, model, loss, len(X_val), device=device)

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
        X_test.to(device)
        y_test.to(device)

        outputs = model(X_test)
        correct = np.sum(sigmoid(outputs).array().round() == y_test.array())
        accuracy = 100 * correct / len(y_test)
        print(f"Test Accuracy:{accuracy}")
