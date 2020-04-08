#-*-coding:utf-8-*-
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import yaml
import torchvision.transforms as transforms
class PoseDataset(Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        '''
        :param mode: 可以选择train, test, eval
        :param num: mesh点的数目
        :param add_noise: 是否加入噪声
        :param root: 数据集的根目录
        :param noise_trans: 噪声增强相关参数
        :param refine: 是否需要为refine模型提供相应的数据
        '''
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]    # 目标物体类别序列号与数据集相对应
        self.mode = mode                                                # 模式，可以选择train, test, eval
        self.list_rgb = []                                              # 存储RGB图像的路径
        self.list_depth = []                                            # 存储深度图像的路径
        self.list_label = []                                            # 存储分割出来物体对应的mask
        self.list_obj = []                                              # 和list_rank一起获得图片的路径，物体类别和图片下标
        self.list_rank = []                                             # 从txt文件读取到的和list_obj一起获得图片的路径，物体类别和图片下标
        self.meta = {}                                                  # 矩阵信息，拍摄图片时的旋转矩阵和偏移矩阵以及物体box
        self.pt = {}                                                    # 保存目标模型点云数据，及.ply文件中的数据
        self.root = root                                                # 数据所在目录
        self.noise_trans = noise_trans                                  # 噪声增强相关参数
        self.refine = refine                                            # 是否需要为refine模型提供相应数据
        item_count = 0                                                  # 计数器(记录处理图片的数目)
        for item in self.objlist:                                       # 对每个目标物体的相关数据进行处理
            # 根据训练或者测试获得相应文件中的txt内容，其中保存的都是图片对应的名称数目
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            # 循环txt记录的每张图片进行处理
            while 1:
                # 记录处理的数目
                item_count += 1
                input_line = input_file.readline()
                # test模式下，图片序列不为10的倍数则continue
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                # 文件读取完成
                if not input_line:
                    break
                if input_line[-1:]  == '\n':
                    input_line = input_line[:-1]
                # 把RGB图像的路径加载到self.list_rgb列表中
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 把depth图像的路径加载到self.list_depth列表中
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 如果是评估模式，则添加segnet_results图片的mask,否则添加data中的mask图片(该为标准mask)
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 把物体的下标,和txt读取到的图片数目标记分别添加到list_obj和list_rank中
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))
            # gt.yml保存拍摄图片时，物体的旋转矩阵以及偏移矩阵，以及物体标签的box
            # 通过该参数,把对应的图片从2维空间恢复到3维空间
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            self.meta[item] = yaml.load(meta_file)
            # 保存目标物体，拍摄物体第一帧的点云数据，成为模型数据
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            print('Object {0} buffer loaded'.format(item))
        self.length = len(self.list_rgb)                                # 读取所有图片的路径，以及文件信息之后，打印图片数目
        self.cam_cx = 325.26110                                         # 摄像头中心横坐标
        self.cam_cy = 242.04899                                         # 摄像头中心纵坐标
        self.cam_fx = 572.41140                                         # 摄像头x轴长度
        self.cam_fy = 573.57043                                         # 摄像头y轴长度
        # 图片的x和y坐标
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])     # xmap.shape=[480, 640]
        '''
        xmap:
        [[  0   0   0 ...   0   0   0]
        [  1   1   1 ...   1   1   1]
        [  2   2   2 ...   2   2   2]
        ...
        [477 477 477 ... 477 477 477]
        [478 478 478 ... 478 478 478]
        [479 479 479 ... 479 479 479]]
        '''
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])     # ymap.shape=[480, 640]
        '''
        [[  0   1   2 ... 637 638 639]
        [  0   1   2 ... 637 638 639]
        [  0   1   2 ... 637 638 639]
        ...
        [  0   1   2 ... 637 638 639]
        [  0   1   2 ... 637 638 639]
        [  0   1   2 ... 637 638 639]]
        '''
        self.num = num                                                  # 获取目标物体点云的数据
        self.add_noise = add_noise                                      # 是否加入噪声
        # 修改输入图像的4大参数值: brightness, contrast and saturation, hue(亮度, 对比度, 饱和度和色度)
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        # 进行标准化
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 边界列表，将一个图片切割成多个坐标
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500                                    # 点云的最大数目
        self.num_pt_mesh_small = 500                                    # 点云的最小数目
        self.symmetry_obj_idx = [7, 8]                                  # 对称物体的标号，实际上对应的是10和11号物体

def ply_vtx(path):
    return 1