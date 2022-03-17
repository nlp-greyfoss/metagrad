from examples.feedforward import load_dataset
import metagrad.module as nn

from metagrad.dataloader import DataLoader
from metagrad.dataset import TensorDataset
from metagrad.functions import sigmoid
from metagrad.loss import BCELoss
from metagrad.optim import SGD
from metagrad.tensor import no_grad, Tensor
from metagrad.utils import Animator, run_epoch, regression_classification_metric


class DynamicFFN(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        '''

        :param num_layers: 隐藏层层数
        :param input_size: 输入维度
        :param hidden_size: 隐藏层大小
        :param output_size: 分类个数
        '''
        layers = []

        layers.append(nn.Linear(input_size, hidden_size))  # 隐藏层，将输入转换为隐藏向量
        layers.append(nn.ReLU())  # 激活函数

        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size // 2))
            hidden_size = hidden_size // 2 # 后面的神经元数递减
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))  # 输出层，将隐藏向量转换为输出

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_model(model, train_dl, val_dl, num_epochs=20):
    opt = SGD(model.parameters(), lr=0.1)
    loss = BCELoss(reduction=None)

    val_losses = []

    for epoch in range(num_epochs):
        train_loss, _ = run_epoch(model, train_dl, loss, opt, activate_func=sigmoid,
                                  evaluate_func=regression_classification_metric)
        with no_grad():
            val_loss, _ = run_epoch(model, val_dl, loss, opt=None, activate_func=sigmoid,
                                    evaluate_func=regression_classification_metric)
        val_losses.append(val_loss)
        print(f'epoch:{epoch + 1}, train loss:{train_loss:.4f}, validation loss:{val_loss:.4f}')

    return val_losses


def compare_model(train_dl, val_dl, simple_model, complex_model, simple_label='Simple model',
                  complex_label='Complex model'):
    num_epochs = 20
    print(f'Training {simple_label}:')
    original_losses = train_model(simple_model, train_dl, val_dl, num_epochs)
    print(f'Training {complex_label}:')
    new_losses = train_model(complex_model, train_dl, val_dl, num_epochs)

    animator = Animator(xlabel='epoch', ylabel='validation loss', yscale='log',
                        xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                        legend=[simple_label, complex_label], saved_file='animator')
    for epoch in range(num_epochs):
        animator.add(epoch + 1, (original_losses[epoch], new_losses[epoch]))

    animator.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X_val, y_val = load_dataset()

    batch_size = 512
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size)

    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    input_size = 10000
    output_size = 1

    simple_model = DynamicFFN(1, input_size, 4, output_size)
    complex_model = DynamicFFN(4, input_size, 128, output_size)

    compare_model(train_dl, val_dl, simple_model, complex_model)
