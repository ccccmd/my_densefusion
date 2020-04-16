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

    knn = KNearestNeighbor(1)                                           # 加快计算速度，利用C++库进行编辑
    bs, num_p, _ = pred_c.size()                                        # 返回bs, 500
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))    # 进行正则化，返回的结果为[bs, 500, 4]
    # base[500, 3, 3], 把预测的旋转参数转换为旋转矩阵
    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1),\
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1),\
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1),\
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
    ori_base = base                                                     # 记录摄像头旋转矩阵
    base = base.contiguous().transpose(2, 1).contiguous()               # 转换维度[500, 3, 3]
    # [500, 500, 3]     每个点云需要与所有的predicted点云做距离差
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    # [500, 500, 5]     每个点云需要与所有的predicted点云做距离差
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target                                                 # 记录初始的目标点云
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t                                                      # 记录原始预测的偏移矩阵


