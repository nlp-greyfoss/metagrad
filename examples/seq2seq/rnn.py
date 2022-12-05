import random
from typing import Tuple

import numpy

from examples.embeddings.utils import BOS_TOKEN, PAD_TOKEN
from examples.seq2seq.base import Encoder, Decoder
import metagrad.module as nn
from examples.seq2seq.dataset import load_dataset_nmt
from metagrad import Tensor, cuda
import metagrad.functions as F
from metagrad.loss import CrossEntropyLoss
from metagrad.optim import Adam, SGD
from metagrad.tensor import no_grad
from metagrad.utils import Animator, Accumulator, Timer
from tqdm import tqdm


class RNNEncoder(Encoder):
    '''用RNN实现编码器'''

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=.0, **kwargs):
        super(RNNEncoder, self).__init__(**kwargs)
        # 嵌入层 获取输入序列中每个单词的嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 基于单向GRU实现
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout, reset_parameters=False)

    def forward(self, X) -> Tensor:
        '''

        Args:
            X:  形状 (batch_size, num_steps)
        Returns:

        '''

        X = self.embedding(X)  # X 的形状 (batch_size, num_steps, embed_size)
        X = X.permute((1, 0, 2))  # (num_steps, batch_size, embed_size)
        output, state = self.rnn(X)
        # num_directions = 1 todo 可以试试num_directions=2
        # output (num_steps, batch_size, num_hiddens * num_directions)
        # state (num_layers * num_directions, batch_size, num_hiddens)
        return output, state


# encoder = RNNEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# encoder.eval()
# X = Tensor.zeros((4, 7), dtype=numpy.int32)
# output, state = encoder(X)
#
#
# print(output.shape)


class RNNDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=.0, **kwargs):
        super(RNNDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # embed_size + num_hiddens 为了处理拼接后的维度，见forward函数中的注释
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, batch_first=False, dropout=dropout,
                          reset_parameters=False)
        # 将隐状态转换为词典大小维度
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.vocab_size = vocab_size

    def init_state(self, enc_outputs):
        return enc_outputs[1]  # 得到编码器输出中的state

    def forward(self, X, state) -> Tuple[Tensor, Tensor]:
        '''

        Args:
            X: (batch_size)
            state:

        Returns:

        '''
        X = X.unsqueeze(0)  # (1, batch_size)
        X = self.embedding(X)  # (1, batch_size, embed_size)

        # context 形状 (1, batch_size, num_hiddens )
        context = state[-1].unsqueeze(0)
        # 为了每个解码时间步都能看到上下文，拼接context与X
        # (1, batch_size, embed_size) + (1, batch_size, num_hiddens)
        #                           => (1, batch_size, embed_size + num_hiddens)
        concat_context = F.cat((X, context), 2)

        output, state = self.rnn(concat_context, state)

        output = self.dense(output.squeeze(0))  # (batch_size, vocab_size)

        return output, state


# decoder = RNNDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# decoder.eval()
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print(output.shape, state.shape)


class Seq2Seq(nn.Module):
    '''合并编码器和解码器'''

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, tf_ratio=0.5) -> Tensor:
        '''

        Args:
            src: (batch_size, num_steps)
            tgt: (batch_size, num_steps)
            tf_ratio: teacher forcing 阈值

        Returns:

        '''
        enc_outputs = self.encoder(src)
        hidden = self.decoder.init_state(enc_outputs)  # (num_layers  * num_directions, batch_size, num_hiddens)

        batch_size, num_steps = tgt.shape

        outputs = []

        inp = Tensor([tgt_vocab[BOS_TOKEN]] * batch_size, device=self.device)  # bos
        for t in range(1, num_steps):
            output, hidden = self.decoder(inp, hidden)

            outputs.append(output)
            tf = random.random() < tf_ratio
            pred = output.argmax(axis=1)
            inp = tgt[:, t] if tf else pred

        outputs = F.stack(outputs, axis=0)
        return outputs


class MaskedSoftmaxCELoss(CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, padding_value):
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred, label)
        label_valid = label != padding_value
        weighted_loss = (unweighted_loss * label_valid).sum()
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    net.to(device)
    optimizer = SGD(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])

    PAD_IDX = tgt_vocab[PAD_TOKEN]

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0

        for batch in data_iter:
            optimizer.zero_grad()
            X, Y = [x.to(device) for x in batch]
            Y_hat = net(X, Y)
            Y = Y[:, 1:].view(-1)
            output_dim = Y_hat.shape[-1]
            Y_hat = Y_hat.view(-1, output_dim)
            l = loss(Y_hat, Y, PAD_IDX)

            l.backward()
            optimizer.step()

            epoch_loss += l.item()

        # if (epoch + 1) % 5 == 0:
        print(f'{epoch} : loss {epoch_loss / len(data_iter) :.3f}')
        #   animator.add(epoch + 1, epoch_loss / len(data_iter))

    print(f'loss {epoch_loss / len(data_iter) :.3f}')
    # animator.show()


def predict_seq2seq(net, data_iter, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    output = []
    with no_grad():
        for batch in data_iter:
            output_seq = []

            X, Y = [x.to(device) for x in batch]

            enc_outputs = net.encoder(X)
            hidden = net.decoder.init_state(enc_outputs)  # (num_layers  * num_directions, batch_size, num_hiddens)

            inp = Tensor([tgt_vocab[BOS_TOKEN]], device=device)
            for t in range(1, num_steps):
                hat_y, hidden = net.decoder(inp, hidden)

                inp = hat_y.argmax(axis=1)
                pred = inp.squeeze(axis=0).item()

                if pred == tgt_vocab['<eos>']:
                    break
                output_seq.append(pred)

            output.append(
                f" {' '.join(src_vocab.token(X.squeeze().data.tolist()))} => {''.join(tgt_vocab.token(output_seq))}")

    return output


# 参数定义
embed_size = 128
num_hiddens = 256
num_layers = 2
dropout = 0.2

batch_size = 64
num_steps = 20

lr = 0.001
num_epochs = 1
min_freq = 1
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

# 加载数据集
train_iter, src_vocab, tgt_vocab = load_dataset_nmt('../data/en-cn/train.txt', batch_size=batch_size,
                                                    min_freq=min_freq, max_len=num_steps)

# 构建编码器
encoder = RNNEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
# 构建解码器
decoder = RNNDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
# 编码器-解码器
net = Seq2Seq(encoder, decoder, device)
# 训练
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

test_iter, _, _ = load_dataset_nmt('../data/en-cn/test_mini.txt', batch_size=1, min_freq=min_freq, max_len=num_steps,
                                   src_vocab=src_vocab, tgt_vocab=tgt_vocab, shuffle=False)

output = predict_seq2seq(net, test_iter, src_vocab, tgt_vocab, num_steps=num_steps, device=device)
net.save("net.pt")
print(output)
