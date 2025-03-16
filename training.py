# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import matplotlib as plt
from torch.utils.data import random_split


# 默认参数设置
DEFAULT_DATASET = 'data'
DEFAULT_MODEL = 3  # 选择第GCN
DEFAULT_CUDA = "cuda:0"

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):  # 定义训练函数，参数包括模型，设备，训练集，优化器，迭代次数
    print('Training on {} samples...'.format(len(train_loader.dataset)))  # 打印训练集样本数，用.format()函数
    model.train()  # 首先将模型设置为训练模式
    for batch_idx, data in enumerate(train_loader):  # 用enmerate()函数得到每个批次的数据集和索引，再遍历，这里train_loader不理解可以看下边解释
        data = data.to(device)   # 对于每个批次数据集加载到指定设备，下边也有定义，这里指的是CPU/GPU设备
        optimizer.zero_grad()  # 将优化器的梯度清零
        output = model(data)  # 通过模型计算出这一批次数据集输出
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))   # 首先将data的y值转化为二维浮点型张量并且加载到指定设备中，再通过损失函数计算其损失值，用于计算梯度，更新参数，模型评估，损失函数loss_fu后边有定义
        loss.backward()  # 将损失反向传播
        optimizer.step()  # 更新模型参数
        if batch_idx % LOG_INTERVAL == 0:  # 如果这里定义个参数LOG每LOG批次打印以下如下包括迭代次数、遍历到的样本总数、样本总数、进行的百分比、损失值
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):  # #定于预测函数，参数有模型，设备，数据
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_dynamic_features = torch.Tensor()  # 添加动态特征的存储
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_dynamic_features = torch.cat((total_dynamic_features, data.dynamic_features.cpu()), 0)  # 记录动态特征
        return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_dynamic_features.numpy()  # 返回动态特征


# datasets = [['data'][int(sys.argv[1])]]
# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
# model_st = modeling.__name__

# 设置默认值
datasets = [DEFAULT_DATASET]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][DEFAULT_MODEL]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 打印GPU设备数量
    print('Number of available GPUs:', torch.cuda.device_count())
    # 打印当前GPU设备的名称
    print('GPU device name:', torch.cuda.get_device_name(0))  # 0表示第一个GPU设备
    # 打印CUDA版本
    print('CUDA version:', torch.version.cuda)
else:
    print('CUDA is not available. Training on CPU...')


TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 5 * 1e-4
LOG_INTERVAL = 20
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


def rmse(y, f):
    return np.sqrt(mean_squared_error(y, f))
def pearson(y, f):
    return pearsonr(y, f)[0]

# 预测和计算评估指标
def evaluate_model(model, device, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_dynamic_features = torch.Tensor()  # 添加动态特征的存储
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            total_dynamic_features = torch.cat((total_dynamic_features, data.dynamic_features.cpu()), 0)  # 记录动态特征

    preds = total_preds.numpy().flatten()
    labels = total_labels.numpy().flatten()

    # 计算评估指标
    rmse_value = rmse(labels, preds)
    pr_value = pearson(labels, preds)

    return rmse_value, pr_value

# 在不同训练集上进行迭代
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        # 导入数据
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')

        # 设置随机种子
        torch.manual_seed(42)
        # 计算训练集和验证集的大小
        total_size = len(train_data)
        val_size = int(0.2 * total_size)  # 20% 用作验证集
        train_size = total_size - val_size

        # 随机划分训练集和验证集
        train_subset, val_subset = random_split(train_data, [train_size, val_size])

        # 创建数据加载器
        train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)  # 验证集不需要打乱
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)


        # 训练模型
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_rmse = 1e8
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        tmp = 0
        rmse_test = 0
        pr_test = 0
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)
            # 评估模型
            rmse_value, pr_value = evaluate_model(model, device, val_loader)
            if rmse_value < best_rmse:
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_file_name)  # 保存当前最好的模型
                print('RMSE improved at epoch ', best_epoch, '; Best RMSE:', rmse_value, '; Pr:',pr_value)
                tmp = pr_value
                best_rmse = rmse_value
                rmse_test, pr_test = evaluate_model(model, device, test_loader)
            elif rmse_value == best_rmse and tmp < pr_value:
                tmp = pr_value
                torch.save(model.state_dict(), model_file_name)  # 保存当前最好的模型
                rmse_test, pr_test = evaluate_model(model, device, test_loader)
            else:
                print('No improvement since epoch ', best_epoch, '; Best RMSE:', best_rmse, '; Pr:',tmp)

        print('Best RMSE achieved at epoch {}: {:.6f}'.format(best_epoch, best_rmse))
        print('RMSE_test', rmse_test)
        print('Pr_test', pr_test)
