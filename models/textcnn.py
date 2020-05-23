# -*- coding: utf-8 -*-
"""
TextCNN 基本做法：

注：输入数据未二维 （batch, seq_len）

1、将输入数据进行向量化编码(Embedding， 此时向量:[batch, seq_len, embed_dim]),
2、然后再将编码后的向量每分别用3个卷积进行卷积操作
3、将3个卷积的结构进行拼接
4、解全连接进行分类操作

TextCNN速度快，简单特征的文本分类效果不错。
注意：
    卷积层输出特征层为2层的时候，效果会比较差，增加到更多层（比如256）效果会好很多。
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """ TextCNN文本分类网络

    Parameters
    ----------
    vacob_size :  词典大小

    n_class: 文本类别

    embed_dim: Embedding的维度


    网络构成:
    Input:     [batch, seq_len]            输入数据，bacth步长，seq_len句子长度(输入为文本词语/字的索引)
    Embedding: [batch, seq_len, embed_dim]  对索引编写编码（向量化过程）
    unsqueeze: [batch, 1, seq_len, embed_dim]
    3个卷积:    [batch, 2, se1_len-f+1, 1] x 3       采用最大池化替代平均和操作
    3个池化:    [batch, 2, 1, 1] x 3
    squeeze:   [batch, 2] x 3
    cat:       [batch, 6]
    Linear:    [batch, n_class]       全连接，分类器


    Examples
    ------------
    # 文本多分类

    vacob_size = 5000;
    seq_len = 50;
    n_class = 10
    x_ = torch.ones((32, seq_len)).to(torch.int64)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = TextCNN(vacob_size, n_class)
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
                 num_filters=256,  # 等于2的效果会比较差，等于256的效果会比较好
                 ):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (f, embed_dim)) for f in [2, 3, 4]])
        self.fc = nn.Linear(in_features=num_filters * 3,
                            out_features=n_class)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.unsqueeze(1)     # [batch, 1, seq_len, embed_dim]   增加通道维度
        pooled = []
        for conv in self.convs:
            out = conv(x)      # [batch, 2, seq_len-f+1, 1]
            out = F.relu(out)
            out = F.max_pool2d(out, (out.shape[-2], 1)) # [batch, 2, 1, 1]
            out = out.squeeze()  # [batch, 2]
            pooled.append(out)
        x = torch.cat(pooled, dim=-1)  # [batch, 6]
        x = self.fc(x)    # [batch, n_class]
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':

    vocab_size = 5000
    seq_len = 50
    n_class = 10
    x_ = torch.ones((32, seq_len)).to(torch.int64)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = TextCNN(vocab_size, n_class)
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










