# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import numpy as np
import pickle as pkl

from public.path import thuc_vocab_path, thuc_train_path, thuc_dev_path, thuc_class_path

np.random.seed(1)

__all__ = ['get_vocab', 'get_classs', 'load_dataset', 'dataIter']


def get_vocab(file_path=thuc_train_path, max_chars=10000, min_freq=1, save_path=thuc_vocab_path):
    """
    获取训练数据的所有词汇
    :param file_path: 获取词汇的文件地址（默认为训练样本文件地址，即从训练样本中获取词汇）
    :param max_chars: 最大的词汇数量，即能保存的最大词汇量（默认为10000）
    :param min_freq:  最小词频，即少于这个词频的词语将会被剔除掉（默认为1）
    :param save_path: 词汇保存地址
    :return: 词汇表的字典 char to index
    """
    if os.path.exists(save_path):
        c2i = pkl.load(open(save_path, 'rb'))
        return c2i
    if not os.path.exists(file_path):
        raise ValueError("文本不存在，请检查")
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.read().split("\n")
        texts = [item.strip() for item in texts if len(item)>0]
    char_dict = {}
    for line in texts:
        line_s = line.split('\t')
        if len(line_s) < 2:
            continue
        context, _ = line_s[0], line_s[1]  # 由于训练集存储格式为 data_text \t lable
        for char in context:
            char_dict[char] = char_dict.get(char, 0) + 1
    char_sort = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
    char_sort = [item for item in char_sort if item[1] >= min_freq]
    char_sort = char_sort[:max_chars]
    chars = ['[PAD]', '[UNK]'] + [item[0] for item in char_sort]
    c2i = {c: i for i, c in enumerate(chars)}
    pkl.dump(c2i, open(save_path, 'wb'))
    return c2i


def get_classs():
    """ 获取各个类型 """
    if not os.path.exists(thuc_class_path):
        raise ValueError("类别文本不存在，请检查！")
    with open(thuc_class_path, 'r', encoding='utf-8') as f:
        texts = f.read().split("\n")
        texts = [str(item).strip() for item in texts if len(str(item).strip()) > 0]
    c2i = {class_: i for i, class_ in enumerate(texts)}
    return c2i


def load_dataset(model='train', max_len=38):
    """
    加载数据集，加载训练或验证数据集的data和label，并将文本转化为index形式
    :param model:  获取数据集的模式，'train'或者'dev'
    :param max_len: 每个样例保持的最长长度
    :return: samples
    """
    if model not in ['train', 'dev']:
        raise ValueError("model 只能是 train、dev其中的一种，请检查")
    if model == 'train':
        file_path = thuc_train_path
    else:
        file_path = thuc_dev_path
    if not os.path.exists(file_path):
        raise ValueError("文本不存在，请检查")
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.read().split("\n")
        texts = [item.strip() for item in texts if len(item)>0]
    c2i = get_vocab()
    pad_id = c2i.get('[PAD]', 0)
    samples = []
    for line in tqdm(texts, desc="字符转index"):
        line_s = line.split('\t')
        if len(line_s) < 2:
            continue
        context, label = line_s[0], line_s[1]
        line_data = ([c2i.get(c, 1) for c in context]) + [pad_id]*(max_len - len(context))
        line_data = line_data[:max_len]
        samples.append((line_data, int(label)))
    samples = np.array(samples)
    return samples


def shuffle_samples(samples):
    """ 打乱数据集 """
    samples = np.array(samples)
    shffle_index = np.arange(len(samples))
    np.random.shuffle(shffle_index)
    samples = samples[shffle_index]
    return samples


def dataIter(samples, batch_size=32, shuffle=True):
    """
    简易数据迭代器
    :param x: 数据数组
    :param y: 标签数组
    :param batch_size: 批次大小
    :param shuffle: 是否打乱数据
    :return: 返回一个数据迭代器
    """
    if shuffle:
        samples = shuffle_samples(samples)

    total = len(samples)
    n_bactch = total//batch_size
    for i in range(n_bactch):
        sub_samples = samples[i*batch_size: (i+1)*batch_size]
        b_x = [item[0] for item in sub_samples]
        b_y = [item[1] for item in sub_samples]
        yield np.array(b_x), np.array(b_y)
    if total%batch_size != 0:  # 处理不够一个批次的数据
        sub_samples = samples[n_bactch * batch_size: total]
        b_x = [item[0] for item in sub_samples]
        b_y = [item[1] for item in sub_samples]
        yield np.array(b_x), np.array(b_y)


class DataIter(object):
    """ 数据迭代器
    说明：Iter 主要需要实现 __next__ 、__iter__、__len__ 三个函数
    """
    def __init__(self, samples, batch_size=32, shuffle=True):
        if shuffle:
            samples = shuffle_samples(samples)
        self.samples = samples
        self.batch_size = batch_size
        self.n_batches = len(samples) // self.batch_size
        self.residue = (len(samples) % self.n_batches != 0)  # 是否为整数
        self.index = 0

    def split_samples(self, sub_samples):
        """ 用于分离data、lable等数据 """
        b_x = [item[0] for item in sub_samples]
        b_y = [item[1] for item in sub_samples]
        return np.array(b_x), np.array(b_y)

    def __next__(self):
        if (self.index == self.n_batches) and (self.residue is True):
            sub_samples = self.samples[self.index*self.batch_size: len(self.samples)]
            self.index += 1
            return self.split_samples(sub_samples)
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            sub_samples = self.samples[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return self.split_samples(sub_samples)

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


if __name__ == '__main__':
    samples = load_dataset('dev')
    print(type(samples))
    class2i = get_classs()
    print(class2i)

    # for x, y in dataIter(trian_x, train_y):
    #     pass
    #     print(y.shape)


