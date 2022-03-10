from examples.feedforward import Feedforward, load_dataset
from metagrad.loss import BCELoss
from metagrad.optim import SGD
from metagrad.tensor import no_grad
from metagrad.utils import Animator, run_epoch


def train_model(hidden_size, X_train, y_train, X_val, y_val, num_epochs=20):
    model = Feedforward(10000, hidden_size, 1)

    opt = SGD(model.parameters(), lr=0.001)
    loss = BCELoss(reduction=None)

    batch_size = 64

    val_losses = []

    for epoch in range(num_epochs):
        train_loss, _ = run_epoch(model, X_train, y_train, loss, opt, batch_size)
        with no_grad():
            val_loss, _ = run_epoch(model, X_val, y_val, loss, opt=None)
        val_losses.append(val_loss)
        print(f'epoch:{epoch + 1}, train loss:{train_loss}, validation loss:{val_loss}')

    return val_losses


def compare_model(X_train, y_train, X_val, y_val, new_size, new_label, original_size=128,
                  original_label='Original model'):
    num_epochs = 20

    original_losses = train_model(original_size, X_train, y_train, X_val, y_val, num_epochs)
    new_losses = train_model(new_size, X_train, y_train, X_val, y_val, num_epochs)

    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                        legend=[original_label, new_label], saved_file='animator')
    for epoch in range(num_epochs):
        animator.add(epoch + 1, (original_losses[epoch], new_losses[epoch]))

    animator.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X_val, y_val = load_dataset()

    compare_model(X_train, y_train, X_val, y_val, new_size=64, new_label='Smaller model')
