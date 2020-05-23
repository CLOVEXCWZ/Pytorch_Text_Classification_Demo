# -*- coding: utf-8 -*-
"""
TextRCNN 基本做法：

注：输入数据未二维 （batch, seq_len）

1、将输入数据进行向量化编码(Embedding， 此时向量:[batch, seq_len, embed_dim]),
2、对Embedding进行双向RNN操作将结果与Embedding进行concat拼接操作
3、在seq_len方向上进行全局池化
4、接全连接层进行分类

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    """ TextRCNN文本分类网络

    Parameters
    ----------
    vacob_size :  词典大小
    n_class: 文本类别
    embed_dim: Embedding的维度
    rnn_hidden: rnn的单元个数

    网络构成:
    ---------

    Input:     [batch, seq_len]            输入数据，bacth步长，seq_len句子长度(输入为文本词语/字的索引)
    Embedding: [batch, seq_len, embed_dim]  对索引编写编码（向量化过程）
    gru:       [batch, seq_len, rnn_hidden] 去了output
    concat:    [batch, seq_len, 2*rnn_hidden+embed_dim]       采用最大池化替代平均和操作
    maxpool2d: [bath, 1, 2*rnn_hidden+embed_dim]
    squeeze:   [bath, 2*rnn_hidden+embed_dim]
    Linear:    [batch, n_class]       全连接，分类器

    Examples
    ------------
    vocab_size = 5000
    seq_len = 50
    n_class = 10
    x_ = torch.ones((32, seq_len)).to(torch.int64)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = TextRCNN(vocab_size, n_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    x_ = x_.to(device)
    y_ = y_.to(device)

    outputs = model(x_)
    optim.zero_grad()
    loss_value = loss(outputs, y_)
    loss_value.backward()
    optim.step()
    print(outputs.shape)
    """

    def __init__(self,
                 vocab_size,  # 词典的大小(总共有多少个词语/字)
                 n_class,  # 分类的类型
                 embed_dim=300,  # embedding的维度
                 rnn_hidden=256,
                 ):
        super(TextRCNN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.gru = nn.GRU(input_size=embed_dim,
                          hidden_size=rnn_hidden,
                          bidirectional=True,
                          batch_first=True)
        self.fc = nn.Linear(in_features=embed_dim+2*rnn_hidden,
                            out_features=n_class)

    def forward(self, x):
        x = self.embedding(x)   # [batch, seq_len, embed_dim]
        # output:[batch, seq_len, rnn_hidden]
        # h_n:[1, batch, 2*rnn_hidden]
        output, h_n = self.gru(x)
        x = torch.cat([x, output], dim=-1)  # [batch, seq_len, 2*rnn_hidden+embed_dim]
        x = F.relu(x)
        x = F.max_pool2d(x, (x.shape[1], 1))  # [batch, 1, 2*rnn_hidden+embed_dim]
        x = x.squeeze()  # [batch, 2*rnn_hidden+embed_dim]
        x = self.fc(x)   # [batch, n_class]
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    vocab_size = 5000
    seq_len = 50
    n_class = 10
    x_ = torch.ones((32, seq_len)).to(torch.int64)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = TextRCNN(vocab_size, n_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    x_ = x_.to(device)
    y_ = y_.to(device)

    outputs = model(x_)
    optim.zero_grad()
    loss_value = loss(outputs, y_)
    loss_value.backward()
    optim.step()
    print(outputs.shape)


