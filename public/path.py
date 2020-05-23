# -*- coding: utf-8 -*-
# 定义一些默认文件、文件夹的路径

import os

cur_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件夹地址

data_set_dir = os.path.join(cur_dir, "../datas")          # 数据集文件夹地址
thuc_news_dir = os.path.join(data_set_dir, "THUCNews")      # THUC新闻数据集地址
thuc_train_path = os.path.join(thuc_news_dir, "train.txt")  # THUC新闻训练数据地址
thuc_dev_path = os.path.join(thuc_news_dir, "dev.txt")      # THUC新闻验证数据地址
thuc_class_path = os.path.join(thuc_news_dir, "class.txt")  # THUC新闻类别文件地址
thuc_vocab_path = os.path.join(thuc_news_dir, "vocab.pkl")

vocab_path = os.path.join(data_set_dir, "vocab.txt")        # 词典地址（此处用的是BERT的词典）

log_dir = os.path.join(cur_dir, "log")

def init_dir():
    """ 初始化文件夹地址，将未创建的文件夹进行创建操作 """
    dirs = [data_set_dir, thuc_news_dir, log_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)










