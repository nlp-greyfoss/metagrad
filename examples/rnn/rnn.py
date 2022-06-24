import numpy as np

from tqdm.auto import tqdm

from examples.embeddings.utils import Vocabulary
from metagrad import Tensor, cuda
import metagrad.module as nn
from metagrad.dataloader import DataLoader
from metagrad.dataset import Dataset
from metagrad.loss import NLLLoss
from metagrad.optim import SGD
from metagrad.utils import pad_sequence
from metagrad.tensor import debug_mode, no_grad
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
                 dropout: float):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers, dropout=dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: Tensor, hidden: Tensor = None) -> Tensor:
        embeded = self.embedding(input)
        output, _ = self.rnn(embeded, hidden)
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
hidden_dim = 256
batch_size = 32
num_epoch = 5
n_layers = 2
dropout = 0.5

# 加载数据
train_data, test_data, vocab, pos_vocab = load_treebank()
train_dataset = RNNDataset(train_data)
test_dataset = RNNDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

num_class = len(pos_vocab)

# 加载模型
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
model = RNN(len(vocab), embedding_dim, hidden_dim, num_class, n_layers, dropout)
model.to(device)

# 训练过程
nll_loss = NLLLoss()
optimizer = SGD(model.parameters(), lr=0.1)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets, mask = [x.to(device) for x in batch]
        log_probs = model(inputs)
        loss = nll_loss(log_probs[mask], targets[mask])
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
