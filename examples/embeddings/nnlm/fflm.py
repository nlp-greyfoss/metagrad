import numpy as np
from tqdm.auto import tqdm

import metagrad.functions as F
import metagrad.module as nn
from examples.embeddings.utils import BOS_TOKEN, EOS_TOKEN, load_corpus, save_pretrained
from metagrad import Tensor, cuda
from metagrad.dataloader import DataLoader
from metagrad.dataset import Dataset
from metagrad.loss import NLLLoss
from metagrad.optim import SGD


class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=4):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 插入句首句尾符号
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < window_size:
                continue
            for i in range(window_size, len(sentence)):
                # 模型输入：长为context_size的上文
                context = sentence[i - window_size:i]
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

        self.data = np.asarray(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = Tensor([ex[0] for ex in examples])
        targets = Tensor([ex[1] for ex in examples])
        return inputs, targets


class FeedForwardNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size, hidden_dim):
        # 单词嵌入E : 输入层 -> 嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 词嵌入层 -> 隐藏层
        self.e2h = nn.Linear(window_size * embedding_dim, hidden_dim)
        #  隐藏层 -> 输出层
        self.h2o = nn.Linear(hidden_dim, vocab_size)

        self.activate = F.relu

    def forward(self, inputs) -> Tensor:
        embeds = self.embeddings(inputs).reshape((inputs.shape[0], -1))
        hidden = self.activate(self.e2h(embeds))
        output = self.h2o(hidden)
        log_probs = F.log_softmax(output, axis=1)
        return log_probs


if __name__ == '__main__':
    embedding_dim = 64
    window_size = 2
    hidden_dim = 128
    batch_size = 1024
    num_epoch = 10
    min_freq = 3  # 保留单词最少出现的次数

    # 读取文本数据，构建FFNNLM训练数据集（n-grams）
    corpus, vocab = load_corpus('../../data/xiyouji.txt', min_freq)
    dataset = NGramDataset(corpus, vocab, window_size)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    # 负对数似然损失函数
    nll_loss = NLLLoss()
    # 构建FFNNLM，并加载至device
    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

    model = FeedForwardNNLM(len(vocab), embedding_dim, window_size, hidden_dim)
    model.to(device)

    optimizer = SGD(model.parameters(), lr=0.001)

    total_losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            log_probs = model(inputs)
            loss = nll_loss(log_probs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss:.2f}")
        total_losses.append(total_loss)

    # 保存词向量（model.embeddings）
    save_pretrained(vocab, model.embeddings.weight.data, "ffnnlm.vec")
