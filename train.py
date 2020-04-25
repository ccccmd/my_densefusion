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
from torch.utils.data import DataLoader
from model.PoseNet import PoseNet
from model.PoseRefineNet import PoseRefineNet
from loss.loss import Loss
from loss.loss_refiner import Loss_refine
from setup_logger import setup_logger
from torchsummary import summary

# 用于命令行输入参数

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
opt = parser.parse_args()


def main():
    print('------------')
    opt.manualSeed = random.randint(1, 100)                                     # 设定随机数
    random.seed(opt.manualSeed)                                                 # 设定随机种子
    torch.manual_seed(opt.manualSeed)                                           # 设定随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # 设定设备

    opt.num_objects = 13                                                        # 训练数据的物体种类数目
    opt.num_points = 500                                                        # 输入点云的数目
    opt.outf = 'trained_models/linemod'                                         # 训练模型保存的目录
    opt.log_dir = 'experiments/logs/linemod'                                    # log保存的目录
    opt.repeat_epoch = 20                                                       # 重复epoch数目

    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects)     # 网络构建，构建完成，对物体的6D姿态进行预测
    estimator.to(device)                                                        # 选择设备
    refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects) # 对初步预测的姿态进行提炼
    refiner.to(device)                                                          # 选择设备

    if opt.resume_posenet != '':                                                # 对posenet模型的加载，如果有的话，
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':                                              # 对refinenet模型的加载，如果有的话，
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))

        opt.refine_start = True                                                 # 标记refine网络开始训练
        opt.decay_start = True                                                  # 标记refine网络参数开始衰减
        opt.lr *= opt.lr_rate                                                   # 学习率变化
        opt.w *= opt.w_rate                                                     # 权重衰减率变化
        opt.batch_size = int(opt.batch_size / opt.iteration)                    # batchsize设定

        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)                 # 设定refine的优化器
    else:
        opt.refine_start = False                                                # 标记refine网络未开始训练
        opt.decay_start = False                                                 # 标记refine网络参数未开始衰减
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)               # 设定posenet的优化器

    # 加载训练数据集
    dataset = PoseDataset('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    # 加载验证数据集
    test_dataset = PoseDataset('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    test_dataloder = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()                                       # 设定对称物体列表
    opt.num_points_mesh = dataset.get_num_points_mesh()                         # 设定点云数目
    print('----------Dataset loaded!---------\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber '
          'of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset),
                                                                            opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)                         # loss计算
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)           # refine_loss计算

    best_test = np.Inf                                                          # 初始位置最好的模型,loss无限大
    if opt.start_epoch == 1:                                                    # 开始训练，则删除之前的日志
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()                                                       # 记录开始时间

    # 开始循环迭代，训练模型-----------------------------------!

    for epoch in range(opt.start_epoch, opt.nepoch):
        # 保存开始迭代的log信息
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        # 记录每个epoch时间
        logger.info('Train time {0}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time)) + 'Training started'))

        train_count = 0
        train_dis_avg = 0.0                                                     # 用于计算平均距离

        if opt.refine_start:                                                    # 如果refine模型，已经开始训练
            estimator.eval()
            refiner.train()
        else:
            estimator.train()

        optimizer.zero_grad()                                                   # 优化器清零梯度

        for rep in range(opt.repeat_epoch):                                     # 每次epoch重复训练次数
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target, model_points, idx = data           # 读取数据
                '''
                points: 由深度图计算出来的点云，该点云数据以摄像头为参考坐标
                choose: 所选择点云的索引[bs, 1, 500]
                img: 通过box剪切下来的RGB图像
                target: 根据model_points点云信息，以及旋转偏移矩阵转换过的点云信息[bs, 500, 3]
                model_points: 目标初始帧对应的点云信息
                idx: 训练图片的下标
                '''
                # 将数据放到device上
                points, choose, img, target, model_points, idx = points.to(device), choose.to(device), img.to(device), target.to(device), model_points.to(device), idx.to(device)

                # 进行预测获得预测的姿态，和特征向量
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                '''
                pred_r: 旋转矩阵[bs, 500, 4]
                pred_t: 偏移矩阵[bs, 500, 3]
                pred_c: 置信度[bs, 500, 1]
                '''
                # 计算loss
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

                # 如果已经开始了refiner模型的训练
                if opt.refine_start:
                    for iter in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)          # 进行refiner预测
                        # 计算loss得到dis
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()                                          # dis进行反向传播
                else:
                    loss.backward()                                             # 否则，则对loss进行反向传播

                train_dis_avg += dis.item()                                     # 用于计算平均距离
                train_count += 1

                # log信息存储
                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'
                                .format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time)),
                                        epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()                                            # 优化器更新
                    optimizer.zero_grad()                                       # 优化器梯度清零
                    train_dis_avg = 0.0

                if train_count != 0 and train_count % 1000 == 0:                # 模型保存
                    if opt.refine_start:                                        # 已经开始refine模型
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}.pos_model_current.pth'.format(opt.outf))

        print('------------ epoch {0} train finish -----------'.format(epoch))
        # 进行测试
        logger = setup_logger('epoch%d test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        # 记录测试时间
        logger.info('Test time {0}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time)) + ',' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()                                                        # 验证模型构建
        refiner.eval()                                                          # refiner模型

        for j, data in enumerate(test_dataloder, 0):
            points, choose, img, target, model_points, idx = data  # 读取数据
            '''
            points: 由深度图计算出来的点云，该点云数据以摄像头为参考坐标
            choose: 所选择点云的索引[bs, 1, 500]
            img: 通过box剪切下来的RGB图像
            target: 根据model_points点云信息，以及旋转偏移矩阵转换过的点云信息[bs, 500, 3]
            model_points: 目标初始帧对应的点云信息
            idx: 训练图片的下标
            '''
            # 将数据放到device上
            points, choose, img, target, model_points, idx = points.to(device), choose.to(device), img.to(
                device), target.to(device), model_points.to(device), idx.to(device)

            # 进行预测获得预测的姿态，和特征向量
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            '''
            pred_r: 旋转矩阵[bs, 500, 4]
            pred_t: 偏移矩阵[bs, 500, 3]
            pred_c: 置信度[bs, 500, 1]
            '''

            # 对结果进行评估
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

            # 如果refine模型开始训练，则同样进行评估
            if opt.refine_start:
                for iter in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()                                              # 用于计算平均距离
            # 保存eval的log
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time)), test_count, dis))
            test_count += 1

        test_dis = test_dis / test_count                                        # 计算平均距离
        logger.info('Test time {0} Epoch {1} Test finish avg_dis:{2}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - st_time)), epoch, test_dis))

        if test_dis <= best_test:                                               # 如果此次测试结果最好，则保留当前测试结果
            best_test = test_dis
            if opt.refine_start:                                                # 保存refiner
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print('----------------test model save finished-------------------')

        # 参数变化

        # 判断模型是否达到衰减要求
        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate                                               # 学习率衰减
            opt.w *= opt.w_rate                                                 # 权重衰减
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        # 模型没有达到loss阈值要求，refine_start = False，则修改相关参数，传递相关数，更新dataset和dataloader
        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            # 训练
            dataset = PoseDataset('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            # 测试
            test_dataset = PoseDataset('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            test_dataloder = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()
            print('----------Dataset loaded!---------\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber '
                'of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset),
                                                                                  opt.num_points_mesh, opt.sym_list))
            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)


if __name__ == '__main__':
    main()

