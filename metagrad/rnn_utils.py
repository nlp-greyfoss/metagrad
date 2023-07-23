from typing import NamedTuple, List

from metagrad import Tensor
import metagrad.functions as F


class PackedSequence(NamedTuple):
    data: Tensor  # 包含packed sequence
    batch_sizes: List[int]  # 序列每个时间步的批大小


def pack_padded_sequence(input: Tensor, lengths: List[int], batch_first: bool = False):
    """
    压缩填充后的序列，批次内序列需要先按照有效长度降序排序
    :param input: 输入序列  如果batch_first=True，形状为(batch_size, seq_len, embdding_size)
                          如果batch_first=False，形状为(seq_len, batch_size, embdding_size)
    :param lengths: 批次内每个序列的有效长度
    :param batch_first: 是否批大小维度在前
    :return:
    """
    if batch_first:
        # 转换成seq_len在前的形式
        input = input.transpose((1, 0, 2))

    steps = []
    # 每个step的批大小
    batch_sizes = []
    # 对长度进行逆序
    lengths_iter = reversed(lengths)
    # 当前长度
    current_length = next(lengths_iter)
    # 取出批大小
    batch_size = input.size(1)
    # lengths应该包含批大小个序列
    if len(lengths) != batch_size:
        raise ValueError("lengths array has incorrect size")
    # 现在是seq_len在前的形式，按seq_len维度取出每个句子，索引(step)从1开始
    for step, step_value in enumerate(input, 1):
        steps.append(step_value[:batch_size])  # 把step_value添加到steps，:batch_size取有效数据(不包括填充)
        batch_sizes.append(batch_size)  # 记录该step有效的序列个数

        while step == current_length:  # 表示此时长度为current_length的填完了
            try:
                new_length = next(lengths_iter)  # 按照逆序取新的长度
            except StopIteration:  # 遍历完lengths_iter
                current_length = None  # 将current_length设为None
                break  # 跳出while循环

            batch_size -= 1  # 但批大小减去1
            current_length = new_length  # 新的长度赋值给current_length

        if current_length is None:  # 表示此时已经遍历完了
            break  # 可以跳出for循环

    return PackedSequence(F.cat(steps), batch_sizes)


def pad_packed_sequence(sequence: PackedSequence, batch_first=False):
    """
    pack_padded_sequence的逆操作
    :param sequence: PackedSequence
    :param batch_first: 是否批大小维度在前
    :return:
    """
    # 取出data和batch_sizes
    var_data, batch_sizes = sequence
    # 0位置一定包含最大的批大小
    max_batch_size = batch_sizes[0]
    # 构建一个输出Tensor 形状为 (seq_len, batch_size, hidden_size?)
    output = Tensor.zeros((len(batch_sizes), max_batch_size, *var_data.shape[1:]))
    # 批次内实际的序列长度
    lengths = []
    # data的偏移量
    data_offset = 0
    # 前一个批大小
    prev_batch_size = batch_sizes[0]
    # 遍历batch_sizes,索引从0开始
    for i, batch_size in enumerate(batch_sizes):
        # 第i个位置(seq_len维度)取var_data从data_offset开始到第batch_size个
        output[i, :batch_size] = var_data[data_offset:data_offset + batch_size]
        # 偏移量加上实际取的batch_size
        data_offset += batch_size
        # 上一个batch_size 减去 当前batch_size
        dec = prev_batch_size - batch_size
        # 如果结果大于0
        if dec > 0:
            # 表示有dec个长度为i的序列
            lengths.extend((i,) * dec)
        # 把batch_size赋给prev_batch_size
        prev_batch_size = batch_size
    # 剩下batch_size个长度为i+1的序列
    lengths.extend((i + 1,) * batch_size)
    # 现在是从小到大的顺序，逆序成从大到小
    lengths.reverse()
    # 如果是batch_first，则转回batch_first的形式，因为在pack_padded_sequence中转了一次
    if batch_first:
        output = output.transpose((1, 0, 2))
    return output, lengths
