#-*-coding:utf-8-*-
import torch
import torch.nn as nn


class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        '''

        :param num_points: 点云数目
        :param num_obj: 目标物体种类
        '''
        super(PoseNet, self).__init__()                                         # 继承Module的八大基本参数
        self.num_points = num_points
