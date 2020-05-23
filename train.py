# -*- coding: utf-8 -*-

from sklearn import metrics
import numpy as np
import torch
import logging
import sys


def train(model,  # 需要训练的模型
          train_iter,  # 训练数据迭代器）
          dev_iter=None,  # 验证数据迭代器
          epochs=20,  # 训练周期
          stop_patience=3,  # 提前终止周期（当验证集的loss值持续stop_patience个周期未提升时，提前终止训练）
          device=None,
          log=None):  # 把数据写到日志（写入到public的log文件夹中）

    if device is None:  # 初始化设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()

    min_loss_epoch = (None, None)
    stop_flag = False

    model_name = model._get_name()
    tip_str = f"\n{model_name} 开始训练....."
    print(tip_str)
    if log:
        log.info(tip_str)

    for epoch in range(epochs):
        loss_value_list = []
        total_iter = len(train_iter)
        for i, (x_batch, y_batch) in enumerate(train_iter):
            x_batch = torch.LongTensor(x_batch).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            outputs = model(x_batch)
            optimizer.zero_grad()
            loss_value = loss(outputs, y_batch)
            loss_value.backward()
            optimizer.step()
            loss_value_list.append(loss_value.cpu().data.numpy())

            str_ = f"{model_name} 周期:{epoch + 1}/{epochs} 步数:{i + 1}/{total_iter} mean_loss:{np.mean(loss_value_list): .4f}"

            sys.stdout.write('\r' + str_)
            sys.stdout.flush()

            if (i + 1) == total_iter and dev_iter is not None:
                acc_, loss_ = eval(model, dev_iter, device)
                str_ = f" 验证集 loss:{loss_:.4f}  acc:{acc_:.4f}"
                # sys.stdout.flush()
                # sys.stdout.close()
                sys.stdout.write(str_)
                sys.stdout.flush()
                print()
                if log:
                    tip_str = f"训练周期:{epoch + 1}/{epochs} 训练集 loss:{np.mean(loss_value_list):.4f} 验证集 loss:{loss_:.4f} acc:{acc_:.4f}"
                    log.info(tip_str)

                model.train()

                if (min_loss_epoch[0] is None) or (min_loss_epoch[0] > loss_):
                    min_loss_epoch = (loss_, epoch)
                else:
                    if (epoch - min_loss_epoch[1]) >= stop_patience:
                        stop_flag = True
                        break

        if stop_flag is True:
            tip_str = f"训练损失值持续 {stop_patience} 个周期没有提高，停止训练"
            # print(tip_str)
            if log:
                log.info(tip_str)
            break



def eval(model, data_iter, device):
    """

    :param model:
    :param data_iter:
    :param device:
    :return:
    """
    model.eval()
    with torch.no_grad():
        acc_list = []
        loss_list = []
        for x, y in data_iter:
            dev_x_ = torch.LongTensor(x).to(device)
            dev_y_ = torch.LongTensor(y).to(device)
            outputs = model(dev_x_)
            p_ = torch.max(outputs.data, 1)[1].cpu().numpy()
            acc_ = metrics.accuracy_score(y, p_)
            loss_ = torch.nn.CrossEntropyLoss()(outputs, dev_y_)
            acc_list.append(acc_)
            loss_list.append(loss_.cpu().data.numpy())
        return np.mean(acc_list), np.mean(loss_list)


def create_log(path, stream=False):
    """
    获取日志对象
    :param path: 日志文件路径
    :param stream: 是否输出控制台
                False: 不输出到控制台
                True: 输出控制台，默认为输出到控制台
    :return:日志对象
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    if stream:
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.DEBUG)
        logger.addHandler(sh)
    # 设置文件日志
    fh = logging.FileHandler(path, encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


if __name__ == '__main__':
    from dataprocess import get_vocab, load_dataset, DataIter
    from models.fasttext import FastText

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    c2i = get_vocab()
    model = FastText(vocab_size=len(c2i), n_class=10)

    # data, labels = load_dataset('train')
    dev_samples = load_dataset('dev')

    trian_iter = DataIter(dev_samples)
    test_iter = DataIter(dev_samples)

    train(model, trian_iter, test_iter, device=device)

    pass





