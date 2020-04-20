#-*-coding:utf-8-*-
from torch.nn.modules.loss import _Loss
import torch
from loss.knn.__init__ import KNearestNeighbor

def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    '''
    和loss类似，差别在没有了置信度，w，refine
    :param pred_r: 预测的旋转参数[bs, 500, 4],相对于摄像头
    :param pred_t: 预测的偏移指数[bs, 500, 3],相对于摄像头
    :param target: 目标姿态，预测的对应图片，通过标准偏移矩阵，结合model_points求得图片对应的点云数据[bs, 500, 3]，点云数据，学习的目标数据
    :param model_points:
    :param idx:
    :param points:
    :param num_point_mesh:
    :param sym_list:
    :return:
    '''