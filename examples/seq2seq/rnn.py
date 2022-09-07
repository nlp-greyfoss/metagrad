from typing import Tuple

import numpy

from examples.embeddings.utils import BOS_TOKEN, PAD_TOKEN
from examples.seq2seq.base import Encoder, Decoder, EncoderDecoder
import metagrad.module as nn
from examples.seq2seq.dataset import load_dataset_nmt
from metagrad import Tensor, cuda
import metagrad.functions as F
from metagrad.loss import CrossEntropyLoss
from metagrad.optim import SGD
from metagrad.tensor import no_grad
from metagrad.utils import Animator, Accumulator, Timer
from tqdm import tqdm


class RNNEncoder(Encoder):
    '''用RNN实现编码器'''

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=.0, **kwargs):
        super(RNNEncoder, self).__init__(**kwargs)
        # 嵌入层 获取输入序列中每个单词的嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 基于GRU实现
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X) -> Tensor:
        '''

        Args:
            X:  形状 (batch_size, num_steps)
        Returns:

        '''

        X = self.embedding(X)  # X 的形状 (batch_size, num_steps, embed_size)
        X = X.permute((1, 0, 2))  # (num_steps, batch_size, embed_size)
        output, state = self.rnn(X)
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
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, batch_first=False, dropout=dropout)
        # 将隐状态转换为词典大小维度
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs):
        return enc_outputs[1]  # 得到编码器输出中的state

    def forward(self, X, state) -> Tuple[Tensor, Tensor]:
        X = self.embedding(X).permute((1, 0, 2))  # (num_steps, batch_size, embed_size)

        # 将最顶层的上下文向量广播成与X相同的时间步，其他维度上只复制1次(保持不变)
        # 形状 (num_layers, batch_size, num_hiddens ) => (num_steps, batch_size, num_hiddens)
        context = state[-1].repeat((X.shape[0], 1, 1))
        # 为了每个解码时间步都能看到上下文，拼接context与X
        # (num_steps, batch_size, embed_size) + (num_steps, batch_size, num_hiddens)
        #                           => (num_steps, batch_size, embed_size + num_hiddens)
        concat_context = F.cat((X, context), 2)

        output, state = self.rnn(concat_context, state)
        output = self.dense(output).permute((1, 0, 2))  # (batch_size, num_steps, vocab_size)

        return output, state


# decoder = RNNDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# decoder.eval()
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print(output.shape, state.shape)


class MaskedSoftmaxCELoss(CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, padding_value):
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred, label)
        label_valid = label != padding_value

        weighted_loss = unweighted_loss * label_valid
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    net.to(device)
    optimizer = SGD(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in tqdm(range(num_epochs)):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，单词数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, Y = [x.to(device) for x in batch]
            bos = Tensor([tgt_vocab[BOS_TOKEN]] * Y.shape[0], device=device).reshape((-1, 1))
            dec_input = F.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input)
            Y = Y.view(-1)
            output_dim = Y_hat.shape[-1]
            Y_hat = Y_hat.view(-1, output_dim)

            l = loss(Y_hat, Y, tgt_vocab[PAD_TOKEN])
            l.sum().backward()
            num_tokens = (Y != 0).sum()
            optimizer.step()
            with no_grad():
                metric.add(l.sum().item(), num_tokens.item())

        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
            print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
                  f'tokens/sec on {str(device)}')

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
    animator.show()


# 参数定义
embed_size = 32
num_hiddens = 32
num_layers = 2
dropout = 0.1

batch_size = 64
num_steps = 20

lr = 0.005
num_epochs = 300
min_freq = 2
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")

# 加载数据集
train_iter, src_vocab, tgt_vocab = load_dataset_nmt('../data/en-cn/train_mini.txt', batch_size=batch_size,
                                                    min_freq=min_freq, max_len=num_steps)
# 构建编码器
encoder = RNNEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
# 构建解码器
decoder = RNNDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
# 编码器-解码器
net = EncoderDecoder(encoder, decoder)
# 训练
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
