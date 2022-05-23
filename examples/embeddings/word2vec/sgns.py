import numpy as np
from tqdm import tqdm

from examples.embeddings.utils import save_pretrained, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_corpus
from metagrad import Tensor, cuda
from metagrad.dataloader import DataLoader
from metagrad.optim import SGD
from metagrad.tensor import debug_mode
from metagrad.dataset import Dataset
import metagrad.functions as F
import metagrad.module as nn


class SGNSDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=2, n_negatives=5, ns_dist=None):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                # 模型输入：(w, context)
                # 输出：0/1，表示context是否为负样本
                w = sentence[i]
                left_context_index = max(0, i - window_size)
                right_context_index = min(len(sentence), i + window_size)
                context = sentence[left_context_index:i] + sentence[i + 1:right_context_index + 1]
                context += [self.pad] * (2 * window_size - len(context))
                self.data.append((w, context))

        # 负样本数量
        self.n_negatives = n_negatives
        # 负采样分布：若参数ns_dist为None，则使用uniform分布
        self.ns_dist = ns_dist if ns_dist is not None else Tensor.ones(len(vocab))

        self.data = np.asarray(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        words = Tensor([ex[0] for ex in examples])
        contexts = Tensor([ex[1] for ex in examples])

        batch_size, window_size = contexts.shape
        neg_contexts = []
        # 对batch内的样本分别进行负采样
        for i in range(batch_size):
            # 保证负样本不包含当前样本中的context
            ns_dist = self.ns_dist.index_fill_(0, contexts[i], .0)
            neg_contexts.append(Tensor.multinomial(ns_dist, self.n_negatives * window_size, replace=True))
        neg_contexts = F.stack(neg_contexts, axis=0)
        return words, contexts, neg_contexts


class SGNSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        # 目标词嵌入
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文嵌入
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_words, pos_contexts, neg_contexts) -> Tensor:
        '''
        word2vec模型比较特殊，我们不关心模型的输出，而是它学到的权重
        为了简单，我们这里直接输出损失
        '''
        batch_size = target_words.shape[0]
        n_negatives = neg_contexts.shape[-1]

        word_embeds = self.w_embeddings(target_words)  # (batch_size, embedding_dim)
        context_embeds = self.c_embeddings(pos_contexts)  # (batch_size, window_size * 2, embedding_dim)
        neg_context_embeds = self.c_embeddings(neg_contexts)  # (batch_size, window_size * n_negatives, embedding_dim)

        word_embeds = word_embeds.unsqueeze(2)

        # 正样本的对数似然
        context_loss = F.logsigmoid((context_embeds @ word_embeds).squeeze(2))
        context_loss = context_loss.mean(axis=1)
        # 负样本的对数似然
        neg_context_loss = F.logsigmoid((neg_context_embeds @ word_embeds).squeeze(axis=2).neg())
        neg_context_loss = neg_context_loss.reshape((batch_size, -1, n_negatives)).sum(axis=2)
        neg_context_loss = neg_context_loss.mean(axis=1)

        # 总损失： 负对数似然
        loss = -(context_loss + neg_context_loss).mean()

        return loss


def get_unigram_distribution(corpus, vocab_size):
    # 从给定语料中统计unigram概率分布
    token_counts = Tensor([.0] * vocab_size)
    total_count = .0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_counts[token] += 1
    unigram_dist = token_counts / total_count
    return unigram_dist


if __name__ == '__main__':
    embedding_dim = 64
    window_size = 2
    batch_size = 10240
    num_epoch = 10
    min_freq = 3  # 保留单词最少出现的次数
    n_negatives = 10  # 负采样数

    # 读取数据
    corpus, vocab = load_corpus('../../data/xiyouji.txt', min_freq)
    # 计算unigram概率分布
    unigram_dist = get_unigram_distribution(corpus, len(vocab))
    # 根据unigram分布计算负采样分数： p(w)**0.75
    negative_sampling_dist = unigram_dist ** 0.75
    # 构建数据集
    dataset = SGNSDataset(corpus, vocab, window_size=window_size, ns_dist=negative_sampling_dist)
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
    model = SGNSModel(len(vocab), embedding_dim)
    model.to(device)

    optimizer = SGD(model.parameters())
    with debug_mode():
        for epoch in range(num_epoch):
            total_loss = 0
            for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
                words, contexts, neg_contexts = [x.to(device) for x in batch]
                optimizer.zero_grad()
                loss = model(words, contexts, neg_contexts)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Loss: {total_loss:.2f}')

    save_pretrained(vocab, model.embeddings.weight, 'sgns.vec')
