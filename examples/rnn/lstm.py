from tqdm.auto import tqdm

import metagrad.functions as F
import metagrad.module as nn
from examples.rnn.rnn import load_treebank, RNNDataset
from metagrad import Tensor, cuda
from metagrad.dataloader import DataLoader
from metagrad.loss import NLLLoss
from metagrad.optim import SGD
from metagrad.tensor import no_grad


class LSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, n_layers: int,
                 dropout: float, bidirectional: bool = False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers, dropout=dropout,
                           bidirectional=bidirectional)

        num_directions = 2 if bidirectional else 1
        self.output = nn.Linear(num_directions * hidden_dim, output_dim)

    def forward(self, input: Tensor, hidden: Tensor = None) -> Tensor:
        embeded = self.embedding(input)
        output, _ = self.rnn(embeded, hidden)  # pos tag任务利用的是包含所有时间步的output
        outputs = self.output(output)
        log_probs = F.log_softmax(outputs, axis=-1)
        return log_probs


embedding_dim = 128
hidden_dim = 128
batch_size = 32
num_epoch = 10
n_layers = 2
dropout = 0.2

# 加载数据
train_data, test_data, vocab, pos_vocab = load_treebank()
train_dataset = RNNDataset(train_data)
test_dataset = RNNDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

num_class = len(pos_vocab)

# 加载模型
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class, n_layers, dropout, bidirectional=True)
model.to(device)

# 训练过程
nll_loss = NLLLoss()
optimizer = SGD(model.parameters(), lr=0.1)

model.train()  # 确保应用了dropout
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets, mask = [x.to(device) for x in batch]
        log_probs = model(inputs)
        loss = nll_loss(log_probs[mask], targets[mask])  # 通过bool选择，mask部分不需要计算
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

# 测试过程
acc = 0
total = 0
model.eval()  # 不需要dropout
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, targets, mask = [x.to(device) for x in batch]
    with no_grad():
        output = model(inputs)
        acc += (output.argmax(axis=-1).data == targets.data)[mask.data].sum().item()
        total += mask.sum().item()

# 输出在测试集上的准确率
print(f"Acc: {acc / total:.2f}")
