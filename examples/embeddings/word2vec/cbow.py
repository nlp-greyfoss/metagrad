import numpy as np
from tqdm import tqdm

import metagrad.module as nn
from metagrad import Tensor, cuda
from metagrad.dataloader import DataLoader
from metagrad.dataset import Dataset
from metagrad.init import uniform_
from metagrad.loss import CrossEntropyLoss
from metagrad.optim import SGD
from metagrad.tensor import debug_mode
from examples.embeddings.utils import BOS_TOKEN, EOS_TOKEN, WEIGHT_INIT_RANGE, load_corpus, save_pretrained


class CBOWDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            # 如果句子长度不足以构建(上下文,目标词)训练样本，则跳过
            if len(sentence) < window_size * 2 + 1:
                continue
            for i in range(window_size, len(sentence) - window_size):
                # 分别取i左右window_size个单词
                context = sentence[i - window_size:i] + sentence[i + 1:i + window_size + 1]
                # 目标词：当前词
                target = sentence[i]
                self.data.append((context, target))

        self.data = np.asarray(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collate_fn(examples):
        '''
        自定义整理函数
        :param examples:
        :return:
        '''
        inputs = Tensor([ex[0] for ex in examples])
        targets = Tensor([ex[1] for ex in examples])
        return inputs, targets


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        # 词向量层，即权重矩阵W
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        uniform_(self.embeddings.weight, -WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE)
        # 输出层，包含权重矩阵W'
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        # 得到所有上下文嵌入向量
        embeds = self.embeddings(inputs)
        # 计算均值，得到隐藏层向量，作为目标词的上下文表示
        hidden = embeds.mean(axis=1)
        output = self.output(hidden)
        return output


if __name__ == '__main__':
    embedding_dim = 64
    window_size = 3
    batch_size = 1024
    num_epoch = 100
    min_freq = 3  # 保留单词最少出现的次数

    corpus, vocab = load_corpus('../../data/xiyouji.txt', min_freq)
    # 构建数据集
    dataset = CBOWDataset(corpus, vocab, window_size=window_size)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

    print(f'current device:{device}')

    loss_func = CrossEntropyLoss()
    # 构建模型
    model = CBOWModel(len(vocab), embedding_dim)
    model.to(device)

    optimizer = SGD(model.parameters(), 1)
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss

        print(f'Loss: {total_loss.item():.2f}')

    save_pretrained(vocab, model.embeddings.weight, 'cbow.vec')
