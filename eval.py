#-*-coding:utf-8-*-
import argparse
import numpy as np
import torch
import torch.nn
import yaml
import copy
from loss.knn.__init__ import KNearestNeighbor
from model.PoseNet import  PoseNet
from model.PoseRefineNet import PoseRefineNet
from datasets.dataset import PoseDataset
from torch.utils.data import DataLoader
from loss.loss import Loss
from loss.loss_refiner import Loss_refine
from transformations import quaternion_matrix, quaternion_from_matrix

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', type=str, default='')             # 验证数据根目录
parser.add_argument('--model', type=str, default='')                    # PoseNet网络模型
parser.add_argument('--refine_model', type=str, default='')             # PoseRefineNet网络模型
opt = parser.parse_args()

num_objects = 13                                                        # 测试目标物体的总数目
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]                 # 物体类别编号列表
num_points = 500                                                        # 根据需要测试的RGB-D，随机选择500个点云
iteration = 4                                                           # 迭代次数
bs = 1                                                                  # batch_size
dataset_config_dir = 'Linemod_preprocessed/models/models_info.yml'      # x,y轴起始位置，对应的半径
output_result_dir = 'result'                                            # eval结果输出路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设定设备

knn = KNearestNeighbor(1)                                               # 最近邻

estimator = PoseNet(num_points=num_points, num_obj=num_objects)         # PoseNet网络模型构建
estimator.to(device)

refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)     # PoseRefineNet网络模型构建
refiner.to(device)

estimator.load_state_dict(torch.load(opt.model))                        # PoseNet模型参数加载
refiner.load_state_dict(torch.load(opt.refine_model))                   # PoseRefineNet模型参数加载
estimator.eval()
refiner.eval()

# 以eval模式，加载数据
test_dataset = PoseDataset('eval', num_points, False, opt.dataset_root, 0.0, True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

sym_list = test_dataset.get_sym_list()                                  # 获取对称物体的索引
num_points_mesh = test_dataset.get_num_points_mesh()                    # 500

criterion = Loss(num_points_mesh, sym_list)                             # Loss加载
criterion_refine = Loss_refine(num_points_mesh, sym_list)               # Loss_refine加载

diameter = []                                                           # 存储模型给定的半径标准，与结果对比
meta_file = open(dataset_config_dir, 'r')                               # 读取model.info文件
meta = yaml.load(meta_file)                                             # 加载model.info文件

for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)               # 存储到diameter中

success_count = [0 for i in range(num_objects)]                         # 用于记录每个目标合格的数目
num_count = [0 for i in range(num_objects)]                             # 用于记录每个目标物体测试总数目

fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')    # 日志记录

for i, data in enumerate(test_dataloader, 0):                           # 数据集遍历
    points, choose, img, target, model_points, idx = data               # 读取数据
    '''
    points: 由深度图计算出来的点云，该点云数据以摄像头为参考坐标
    choose: 所选择点云的索引[bs, 1, 500]
    img: 通过box剪切下来的RGB图像
    target: 根据model_points点云信息，以及旋转偏移矩阵转换过的点云信息[bs, 500, 3]
    model_points: 目标初始帧对应的点云信息
    idx: 训练图片的下标
    '''
    if len(points.size()) == 2:                                         # 检测判断
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!'.format(i))
        continue

    # 将数据放到device上
    points, choose, img, target, model_points, idx = points.to(device), choose.to(device), img.to(device), target.to(device), model_points.to(device), idx.to(device)

    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)   # PoseNet预测
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)  # pred_r[1, 500, 1]
    pred_t = pred_t.view(num_points, 1, 3)                              # pred_t[500, 1, 3]
    pred_c = pred_c.view(1, num_points)                                 # pred_c[1, 500]

    how_max, which_max = torch.max(pred_c, 1)

    # 从所有pose中找到最好的那个
    my_r = pred_r[0][which_max[0]].view(-1).to('cpu').data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).to('cpu').data.numpy()
    my_pred = np.append(my_r, my_t)

    # PoseRefineNet循环迭代
    for iter in range(0, iteration):
        # 类型转换 T[1, 500, 3]
        T = torch.from_numpy(my_t.astype(np.float32)).to(device).view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        my_mat = quaternion_matrix(my_r)                                # 将my_r转换成四元数矩阵
        # 类型转换 R[1, 3, 3]
        R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).to(device).view(1, 3, 3)
        # my_mat用于记录上一次迭代的预测结果
        my_mat[0:3, 3] = my_t

        new_points = torch.bmm((points - T), R).contiguous()            # 把points进行逆操作得到new_points

        pred_r, pred_t = refiner(new_points, emb, idx)                  # 通过refiner得到新的pose
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))     # pred_r标准化，[1, 1, 1]
        my_r_2 = pred_r.view(-1).to('cpu').data.numpy()
        my_t_2 = pred_t.view(-1).to('cpu').data.numpy()
        my_mat_2 = quaternion_matrix((my_r_2))                          # 将my_r_2转换成四元数矩阵
        # my_mat_2用于记录当前refiner预测后的结果
        my_mat_2[0:3, 3] = my_t_2                                       # 获得偏移矩阵
        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)           # 转化为四元数矩阵
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)                     # 把当前估算结果赋值送入下次迭代
        my_r = my_r_final                                               # 更新
        my_t = my_t_final                                               # 更新

    model_points = model_points[0].to('cpu').detach().numpy()
    my_r = quaternion_matrix(my_r)[:3, :3]                              # 取出最终预测的旋转矩阵
    pred = np.dot(model_points, my_r.T) + my_t                          # 获得经过姿态变化后的points
    target = target[0].to('cpu').detach().numpy()                       # 获得目标点云数据

    # 针对对称物体
    if idx[0].item in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).to(device).transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).to(device).transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()

    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    # 与model_info进行对比
    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1

# 每个物体测试准确度
for i in range(num_objects):
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()
