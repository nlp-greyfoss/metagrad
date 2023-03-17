import time

import numpy as np

from tqdm.auto import tqdm

from examples.embeddings.utils import Vocabulary
from metagrad import Tensor, cuda, debug_mode, no_grad
import metagrad.module as nn
from metagrad.dataloader import DataLoader
from metagrad.dataset import Dataset
from metagrad.loss import NLLLoss
from metagrad.optim import SGD, Adam
from metagrad.utils import pad_sequence
import metagrad.functions as F


class RNNDataset(Dataset):
    def __init__(self, data):
        self.data = np.asarray(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collate_fn(examples):
        inputs = [Tensor(ex[0]) for ex in examples]
        targets = [Tensor(ex[1]) for ex in examples]
        inputs = pad_sequence(inputs)
        targets = pad_sequence(targets)
        mask = inputs.data != 0
        return inputs, targets, Tensor(mask)


class RNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, n_layers: int,
                 dropout: float, bidirectional: bool = False, mode: str = 'RNN'):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if mode == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout,
                              bidirectional=bidirectional)
        elif mode == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout,
                               bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout,
                              bidirectional=bidirectional)

        print('model:', self.rnn)

        num_directions = 2 if bidirectional else 1
        self.output = nn.Linear(num_directions * hidden_dim, output_dim)

    def forward(self, input: Tensor, hidden: Tensor = None) -> Tensor:
        embeded = self.embedding(input)
        output, _ = self.rnn(embeded, hidden)  # pos tag任务利用的是包含所有时间步的output
        outputs = self.output(output)
        log_probs = F.log_softmax(outputs, axis=-1)
        return log_probs


def load_treebank():
    from nltk.corpus import treebank
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    vocab = Vocabulary.build(sents, reserved_tokens=["<pad>"])

    tag_vocab = Vocabulary.build(postags)

    train_data = [(vocab.to_ids(sentence), tag_vocab.to_ids(tags)) for sentence, tags in
                  zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.to_ids(sentence), tag_vocab.to_ids(tags)) for sentence, tags in
                 zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab


embedding_dim = 128
hidden_dim = 128
batch_size = 32
num_epoch = 100
n_layers = 2
dropout = 0.2

# 加载数据
train_data, test_data, vocab, pos_vocab = load_treebank()
train_dataset = RNNDataset(train_data)
test_dataset = RNNDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

num_class = len(pos_vocab)
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

print(device)

for mode in ["RNN", "GRU", "LSTM"]:
    for optim in [SGD, Adam]:
        if mode == "RNN" and optim is Adam:
            print(f"Cancel for mode {mode} and optim {optim.__name__}")
            break

        # 加载模型
        model = RNN(len(vocab), embedding_dim, hidden_dim, num_class, n_layers, dropout, bidirectional=True, mode=mode)
        model.to(device)

        # 训练过程
        nll_loss = NLLLoss()
        lr = 0.1 if optim is SGD else 1e-3
        optimizer = optim(model.parameters(), lr=lr)
        print(optimizer)

        start = time.time()
        model.train()  # 确保应用了dropout

        # with debug_mode():
        for epoch in tqdm(range(num_epoch)):
            total_loss = 0
            for batch in train_data_loader:
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
        model.to_cpu()
        for batch in tqdm(test_data_loader, desc=f"Testing"):
            inputs, targets, mask = batch
            with no_grad():
                output = model(inputs)
                acc += (output.argmax(axis=-1).data == targets.data)[mask.data].sum().item()
                total += mask.sum().item()

        # 输出在测试集上的准确率
        print(f"{mode} optimized by {optim.__name__} Acc: {acc / total:.2f}")
        print(f'Cost:{(time.time() - start)}')
        #      RNN GRU LSTM
        # Adam 92% 92% 92%
        # SGD  84% 85% 84%

