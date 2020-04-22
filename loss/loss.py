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
    # [500, 500, 3]     复制是因为每个点云需要与所有的predicted点云做距离差
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    # [500, 500, 3]     复制是因为每个点云需要与所有的predicted点云做距离差
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target                                                 # 记录初始的目标点云
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t                                                      # 记录原始预测的偏移矩阵
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)
    pred = torch.add(torch.bmm(model_points, base), points + pred_t)    # pred[500, 500, 3]
    if not refine:                                                      # 如果没有训练
        if idx[0].item() in sym_list:                                   # 如果是对称的物体
            target = target[0].transpose(1, 0).contiguous().view(3, -1) # [500, 500, 3] -> [3, 250000]
            pred = pred.permute(2, 0, 1).contiguous().view(3, -1)       # [500, 500, 3] -> [3, 250000]
            inds = knn(target.unsqueeze(0), pred.unsqueeze(0))          # target每个点云和pred点云进行对比，找到距离最近点云的索引[1, 1, 250000]
            target = torch.index_select(target, 1, inds.view(-1).detach() - 1)  # 从target点云中，根据计算出来的索引进行挑选,[3, 250000]
            target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()   # [500, 500, 3]
            pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()   # [500, 500, 3]

    # 求得预测点云和目标点云的平均距离，把置信度和点云距离关联起来
    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)
    pred_c = pred_c.view(bs, num_p)                                     # pred_c[1, 500]
    how_nax, which_max = torch.max(pred_c, 1)                           # which_max表示索引下标,找到置信度最高的地方
    dis = dis.view(bs, num_p)                                           # dis[1, 500]
    t = ori_t[which_max[0]] + points[which_max[0]]                      # 获取最好的偏移矩阵,相对于model_points
    points = points.view(1, bs * num_p, 3)                              # points[1, 500, 3]
    ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()        # 求得置信度最高的旋转矩阵，相对于摄像头
    ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3) # ori_t[1, 500, 3]

    new_points = torch.bmm((points - ori_t), ori_base).contiguous()     # 根据预测最好的旋转矩阵,求得当前帧对应的点云
    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()  # new_target[1, 500, 3]
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3) # ori_t[1, 500, 3]

    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous() # 根据预测最好的旋转矩阵,求得当前帧对应的点云
    del knn
    return loss, dis[0][which_max[0]], new_points.detach(), new_target.detach()

class Loss(_Loss):
    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_points_mesh = num_points_mesh
        self.sym_list = sym_list


    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_points_mesh, self.sym_list)







