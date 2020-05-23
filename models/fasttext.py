# -*- coding: utf-8 -*-
"""
FastText 基本做法：

注：输入数据未二维 （batch, seq_len）

1、将输入数据进行向量化编码(Embedding， 此时向量:[batch, seq_len, embed_dim]),
2、然后再将编码后的向量每一句话相加成一个向量（第一个维度相加， 不过代码中采用了
   最大池化操作），此时向量[batch, 1, embed_dim]
3、直接连接全连接进行分类

FastText的特点就是快，但是由于其处理方式比较粗糙（Embedding后相加）所以会损失一些特征。
"""


import torch.nn as nn
import torch.nn.functional as F
import torch


class FastText(nn.Module):
    """ FastText文本分类网络

    Parameters
    ----------
    vacob_size :  词典大小

    n_class: 文本类别

    embed_dim: Embedding的维度


    网络构成:
    Input:     [batch, seq_len]            输入数据，bacth步长，seq_len句子长度(输入为文本词语/字的索引)
    Embedding: [batch, seq_len, embed_dim]  对索引编写编码（向量化过程）
    MaxPool2d: [batch, 1, embed_dim]        采用最大池化替代平均和操作
    Linear:    [batch, n_class]             全连接，分类器

    Examples
    ------------
    # 文本多分类

    import numpy as np
    import torch

    vacob_size = 5000; seq_len=50; n_class=10
    x = torch.ones((32, seq_len)).to(torch.int64)
    y = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = FastText(vacob_size, n_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    outputs = model(x)
    optim.zero_grad()
    loss_value = loss(outputs, y)
    loss_value.backward()
    optim.step()
    """
    def __init__(self,
                 vocab_size,  # 词典的大小(总共有多少个词语/字)
                 n_class,     # 分类的类型
                 embed_dim=128,  # embedding的维度
                  ):

        super(FastText, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.fc = nn.Linear(in_features=embed_dim,
                            out_features=n_class)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = F.max_pool2d(x, (x.shape[-2], 1)) # [batch, 1, embed_dim]
        x = x.squeeze()        # [batch, embed_dim]
        x = self.fc(x)         # [batch, n_class]
        x = torch.sigmoid(x)    # [batch, n_class]
        return x


if __name__ == '__main__':

    vacob_size = 5000
    seq_len = 50
    n_class = 10
    x_ = torch.ones((32, seq_len)).to(torch.int64)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = FastText(vacob_size, n_class)
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











