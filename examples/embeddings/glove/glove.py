from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

import metagrad.module as nn
from examples.embeddings.utils import BOS_TOKEN, EOS_TOKEN, load_corpus, save_pretrained
from metagrad import Tensor, cuda
from metagrad.dataloader import DataLoader
from metagrad.dataset import Dataset
from metagrad.optim import SGD


class GloveDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=2):
        # 词与上下文词在语料库中的共现次数
        self.cooccur_counts = defaultdict(float)
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                w = sentence[i]
                left_contexts = sentence[max(0, i - window_size):i]
                right_contexts = sentence[i + 1: min(len(sentence), i + window_size)]
                # 共现次数随距离衰减
                for k, c in enumerate(left_contexts[::-1]):
                    self.cooccur_counts[(w, c)] += 1 / (k + 1)

                for k, c in enumerate(right_contexts):
                    self.cooccur_counts[(w, c)] += 1 / (k + 1)

        data = [(w, c, count) for (w, c), count in self.cooccur_counts.items()]
        self.data = np.asarray(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        words = Tensor([ex[0] for ex in examples], dtype=np.int)
        contexts = Tensor([ex[1] for ex in examples], dtype=np.int)
        counts = Tensor([ex[2] for ex in examples])
        return words.int_(), contexts.int_(), counts


class GloveModel(nn.Module):
    def __init__(self, vocab_size, embeddings_dim, m_max, alpha):
        super(GloveModel, self).__init__()
        # 词嵌入
        self.w_embeddings = nn.Embedding(vocab_size, embeddings_dim)
        # 偏置
        self.w_biases = nn.Embedding(vocab_size, 1)
        # 上下文嵌入
        self.c_embeddings = nn.Embedding(vocab_size, embeddings_dim)
        self.c_biases = nn.Embedding(vocab_size, 1)

        self.m_max = m_max
        self.alpha = alpha

    def forward(self, words, contexts, counts) -> Tensor:
        '''
        类似word2vec，我们这里也直接输出损失
        '''
        w_embeds = self.w_embeddings(words)
        w_biases = self.w_biases(words)

        c_embeds = self.c_embeddings(contexts)
        c_biases = self.c_biases(contexts)

        log_counts = counts.log()
        weight_factor = ((counts / self.m_max) ** self.alpha).clip(x_max=1.0)
        loss = ((w_embeds * c_embeds).sum(axis=1) + w_biases + c_biases - log_counts) ** 2
        weight_avg_loss = (weight_factor * loss).mean()

        return weight_avg_loss


if __name__ == '__main__':
    embedding_dim = 64
    window_size = 2
    batch_size = 1024
    num_epoch = 10
    # 用以控制样本权重的超参数
    m_max = 100
    alpha = 0.75
    min_freq = 3

    # 读取数据
    corpus, vocab = load_corpus('../../data/xiyouji.txt', min_freq)
    # 构建数据集
    dataset = GloveDataset(corpus, vocab, window_size=window_size)
    # 构建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

    print(f'current device:{device}')

    # 构建模型
    model = GloveModel(len(vocab), embedding_dim, m_max, alpha)
    model.to(device)

    optimizer = SGD(model.parameters())
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            words, contexts, counts = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss = model(words, contexts, counts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Loss: {total_loss:.2f}')

    # 合并词嵌入矩阵与上下文嵌入矩阵，作为最终的预训练词向量
    combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight
    save_pretrained(vocab, combined_embeds.data, "glove.vec")
