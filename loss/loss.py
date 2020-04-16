#-*-coding:utf-8-*-
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import torch.nn as nn
import random
from loss.knn.__init__ import KNearestNeighbor

def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
    '''

    :param pred_r: 预测的旋转参数，参考基准是摄像头，[bs, 500, 4]
    :param pred_t: 预测的偏移参数，参考基准是摄像头，[bs, 500, 3]
    :param pred_c: 预测的置信度参数，参考基准是摄像头，[bs, 500, 1]
    :param target: 目标姿态，预测的对应图片，通过标准偏移矩阵，结合model_points求得图片对应的点云数据[bs, 500, 3]，点云数据，学习的目标数据
    :param model_points: 目标模型的点云数据，第一帧的数据[bs, 500, 3]
    :param idx: 训练模型索引
    :param points: 由深度图计算出来的点云数据，参考基准是摄像头
    :param w: learning_rate
    :param refine: 标记是否开始训练refine网络
    :param num_point_mesh: 500，点云数目
    :param sym_list: 对称模型的序列号
    :return: 损失函数的计算结果
    '''
