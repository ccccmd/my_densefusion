#-*-coding:utf-8-*-

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from datasets.dataset import PoseDataset as PoseDataset
from model.PoseNet import PoseNet
from model.PoseRefineNet import PoseRefineNet
from loss.loss import Loss
from loss.loss_refiner import Loss_refine
from torchsummary import summary

# 用于命令行输入参数，后续可能取代------------------------------！

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', type=str, default='')                     # 数据路径
parser.add_argument('--batch_size', type=int, default=8)                        # batch_size
parser.add_argument('--workers', type=int, default=10)                          # 加载数据的线程数目
parser.add_argument('--lr', default=0.0001)                                     # 初始学习率
parser.add_argument('--lr_rate', default=0.3)                                   # 学习率衰减
parser.add_argument('--w', default=0.015)                                       # 初始权重
parser.add_argument('--w_rate', default=0.3)                                    # 权重衰减率
parser.add_argument('--decay_margin', default=0.016)                            # 衰减间隙
parser.add_argument('--refine_margin', default=0.013)                           # 设定loss阈值，表示何时进行refine的训练
parser.add_argument('--noise_trans', default=0.03)                              # 添加噪声
parser.add_argument('--iteration', type=int, default=2)                         # iteration
parser.add_argument('--nepoch', type=int, default=500)                          # epoch
parser.add_argument('--resume_posenet', type=str, default='')                   # 是否训练PoseNet模型，加载预训练模型
parser.add_argument('--resume_refinenet', type=str, default='')                 # 是否训练RefineNet模型，加载预训练模型
parser.add_argument('--start_epoch', type=int, default=1)                       # 初始epoch


def main():
    print('------------')


if __name__ == '__main__':
    main()

