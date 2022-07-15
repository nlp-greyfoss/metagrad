# metagrad

[![GitHub Action](https://github.com/nlp-greyfoss/metagrad/workflows/Unit%20Test/badge.svg)](https://github.com/nlp-greyfoss/metagrad/actions?workflow=Unit%20Test)




* 一个用于学习的仿PyTorch纯Python实现的自动求导工具，参考了PyTorch和tinygrad等优秀开源工具。
* 本着“凡我不能创造的，我就不能理解”的思想，基于纯Python以及NumPy从零创建自己的深度学习框架，该框架类似PyTorch能实现自动求导。
* 从自己可以理解的角度出发，创建一个自己的深度学习框架，让大家切实掌握深度学习底层实现，而不是仅做一个调包侠。
* 核心代码少，适用于教学。



# 实现教程
一步一步实现教程： [从零实现深度学习框架](https://helloai.blog.csdn.net/article/details/122024643)

# 线路图

```mermaid
graph TD
Graph(计算图介绍)  --> Operator(常见运算的计算图)
Operator --> Tensor(实现自己的Tensor对象)
Tensor  --> Softmax回归
Tensor --> LinearRegression(线性回归)
Tensor  -->  LogisticRegression(逻辑回归)
Tensor  -->  逻辑回归数值稳定
Tensor  --> Softmax回归数值稳定
LogisticRegression--> 交叉熵
交叉熵 --> 交叉熵代码优化
交叉熵代码优化 --> 神经元简介
神经元简介--> 过拟合与欠拟合
神经元简介 --> 数据的加载
神经元简介 --> 神经网络入门
神经元简介 --> 理解正则化
神经元简介 --> 权重初始化
神经网络入门 --实战 --> 实现电影评论分类 --> N-Gram语言模型
N-Gram语言模型 --> 利用GPU加速
N-Gram语言模型 --> 从共现矩阵到点互信息
N-Gram语言模型 --> 加速CuPy运算
从共现矩阵到点互信息 --> Word2vec
从共现矩阵到点互信息 --> 神经概率语言模型
从共现矩阵到点互信息 --> GloVe
神经概率语言模型 --> RNN
神经概率语言模型 --> LSTM
神经概率语言模型 --> GRU
 LSTM --> ELMO
ELMO --> Seq2Seq 
ELMO --> 带注意力机制的Seq2seq
ELMO --> 注意力机制
带注意力机制的Seq2seq -- 实战 --> chatbot[基于Seq2seq with Attention 实现聊天机器人] --> CNN
CNN --> TextCNN --> Transformer  --> BERT 

click Graph "https://helloai.blog.csdn.net/article/details/121964022"
click Operator "https://helloai.blog.csdn.net/article/details/122024945"
click Tensor "https://helloai.blog.csdn.net/article/details/122054811"
click LinearRegression "https://helloai.blog.csdn.net/article/details/122241715"
click LogisticRegression "https://helloai.blog.csdn.net/article/details/122446249"
click Softmax回归 "https://helloai.blog.csdn.net/article/details/122546843"
click 逻辑回归数值稳定 "https://helloai.blog.csdn.net/article/details/122610850"
click Softmax回归数值稳定 "https://helloai.blog.csdn.net/article/details/122678759"
click 交叉熵 "https://helloai.blog.csdn.net/article/details/121734499"
click 交叉熵代码优化 "https://helloai.blog.csdn.net/article/details/124546530"
click 神经元简介 "https://helloai.blog.csdn.net/article/details/122775512"
click 过拟合与欠拟合 "https://helloai.blog.csdn.net/article/details/123151877"
click 数据的加载 "https://helloai.blog.csdn.net/article/details/123399307"
click 神经网络入门 "https://helloai.blog.csdn.net/article/details/122949665"
click 理解正则化 "https://helloai.blog.csdn.net/article/details/123716578"
click 权重初始化 "https://helloai.blog.csdn.net/article/details/124332725"
click 实现电影评论分类 "https://helloai.blog.csdn.net/article/details/123036716"
click N-Gram语言模型 "https://helloai.blog.csdn.net/article/details/124384220"
click 利用GPU加速 "https://helloai.blog.csdn.net/article/details/124840221"
click 从共现矩阵到点互信息 "https://helloai.blog.csdn.net/article/details/124872206"
```
