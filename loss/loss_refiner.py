#-*-coding:utf-8-*-
from torch.nn.modules.loss import _Loss
import torch
from loss.knn.__init__ import KNearestNeighbor

def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    '''
    和loss类似，差别在没有了置信度，w，refine
    :param pred_r: 预测的旋转参数,相对于摄像头
    :param pred_t: 预测的偏移指数,相对于摄像头
    :param target: 目标姿态，预测的对应图片，通过标准偏移矩阵，结合model_points求得图片对应的点云数据[bs, 500, 3]，点云数据，学习的目标数据
    :param model_points: 目标模型的点云数据第一帧[bs, 500, 3]
    :param idx: 随机训练的索引
    :param points: 从深度图计算出来的点云，该点云数据以摄像头为参考坐标
    :param num_point_mesh: 500
    :param sym_list: 对称模型的序列号
    :return: 损失函数的计算结果
    '''

    knn = KNearestNeighbor(1)                                           # 加快计算速度，利用C++库进行编辑
    pred_r = pred_r.view(1, 1, -1)                                      # pred_r
    pred_t = pred_t.view(1, 1, -1)                                      # pred_t
    bs, num_p, _ = pred_r.size()                                        # bs = 1, num_p = 1
    num_input_points = len(points[0])                                   # 500

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))    # 对预测的旋转矩阵进行正则化
    # base[1, 1, 3]
    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1),\
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1),\
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1),\
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),\
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).contiguous().view((bs * num_p, 3, 3))

    ori_base = base                                                     # 记录摄像头旋转矩阵
    base = base.contiguous().transpose(2, 1).contiguous()               # 转换维度[bs * num_p, 3, 3]
    # 复制是因为每个点云需要与所有的predicted点云做距离差
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    # 复制是因为每个点云需要与所有的predicted点云做距离差
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target                                                 # 记录初始的目标点云
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)                 # 转换维度[bs * num_p, 1, 3]
    ori_t = pred_t                                                      # 记录原始预测的偏移矩阵

    pred = torch.add(torch.bmm(model_points, base), pred_t)             # 批量矩阵相乘，model_points与旋转矩阵相乘加上偏移矩阵，得到当前帧对应的点云姿态，以model_points为参考

    if idx[0].item() in sym_list:
        target = target[0].transpose(1, 0).contiguous().view(3, -1)     # target转换维度
        pred = pred.permute(2, 0, 1).contiguous().view(3, -1)           # pred转换维度
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))              # target每个点云和pred点云进行对比，找到距离最近点云的索引
        target = torch.index_select(target, 1, inds.view(-1).detach() - 1)  # 从target点云中，根据计算出来的min索引，全部挑选出来
        target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()   # target转换维度
        pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()   # pred转换维度

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)         # 求得预测点云和目标点云的平均距离

    t = ori_t[0]                                                        # 获取偏移矩阵，相对于model_points
    points = points.view(1, num_input_points, 3)                        # points转换维度
    # 根据t和ori_base计算new_points
    ori_base = ori_base[0].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_input_points, 1).contiguous().view(1, bs * num_input_points, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()
    # 根据t和ori_target计算new_target
    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    del knn
    return dis, new_points.detach(), new_target.detach()


class Loss_refine(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        return loss_calculation(pred_r, pred_t, target, model_points, idx, points, self.num_pt_mesh, self.sym_list)





