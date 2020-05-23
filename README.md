# Pytorch 文本(短文本)分类任务 Demo

​	本demo是在学习和练习文本分类的过程中记录下来的一个demo。主要是温习和练习一些基本的文本分类神经网络。文档里面实现的方法基本都有详细的说明，主要是方便后期查看。

​	本demo主要参考**[ Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)** 实现，用到的网络以及数据集都是基于作者的项目的，若想看原滋原味的可点击链接移步到原项目。

**注意和说明**

- 项目主要是练习，所以参数方面并没有过多的调整。
- 关于准确度和提升
  - 项目中模型的参数并未再进行初始化操作
  - 项目词嵌入（dmeo中用的是字嵌入）使用的是随机生成，不过其实采用训练好的词向量进行词嵌入操作效果会更好。
  - [ Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch) 效果更好，主要的原因是：1.网络结构并不是完全一样 2.使用词嵌入初始化方式有差别  3、是否进行网络模型参数初始化 4、数据处理不一样（FastText）

**数据集（摘自[ Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch) )**

我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

| 数据集 | 数据量 |
| ------ | ------ |
| 训练集 | 18万   |
| 验证集 | 1万    |



## **项目结构**

- models
  - textfast.py
  - textcnn.py
  - textrcnn.py
  - textrnn.py
  - transformer.py 
- dataset
  - THUCNews  (数据集)
- public
  - log
    - 日志文件列表（记录训练的数据）
  - path  定义路径 
  - torch_train  模型训练相关
- dataprocess.py  数据处理
- train.py 训练模型相关
- train_all.py  训练所有模型



## 主要用到的库版本

Python ： 3.6

torch :  1.4.0



## 使用说明

可直接运行 train_all.py 即可进行训练

命令行： python  train_all.py 

## 附准确率（训练详细记录文件在 public\log 文件夹内）

| 网络模型    | 准确率 |
| ----------- | ------ |
| FastText    | 85.34% |
| TextCNN     | 89.62% |
| TextRNN     | 88.9%  |
| TextRCNN    | 90.22% |
| Transformer | 88.98% |



