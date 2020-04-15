#-*-coding:utf-8-*-
from torch.utils.data import Dataset
import torch
import numpy as np
import numpy.ma as ma
import yaml
import torchvision.transforms as transforms
from PIL import Image
import cv2
import random

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
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                # 把RGB图像的路径加载到self.list_rgb列表中
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 把depth图像的路径加载到self.list_depth列表中
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 如果是评估模式，则添加segnet_results图片的mask,否则添加data中的mask图片(该为标准mask)
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item,
                                                                                               input_line))
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
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))  # 读取ply文件

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



    def __getitem__(self, index):                                       # dataset的核心关键函数重写

        img = Image.open(self.list_rgb[index])                          # 根据索引获得对应图像的RGB像素
        depth = np.array(Image.open(self.list_depth[index]))            # 根据索引获得对应图像的深度图像素
        label = np.array(Image.open(self.list_label[index]))            # 根据索引获得对应图像的mask像素
        obj = self.list_obj[index]                                      # 获得物体属于的类别的序列号
        rank = self.list_rank[index]                                    # 获得该张图片物体图像的下标

        # 对序列为2的目标物体特殊处理的原因在于只有2的gt.yml文件中的obj_id是不与对应的目标物体序列相对应的，需要进行特殊处理。
        # 不对序列为2的目标物体处理
        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):               # 对该物体的每个图片进行循环
                if self.meta[obj][rank][i]['obj_id'] == 2:              # 将该图片目标为2的赋值给meta
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]                              # 当该物体序列不为2时，将第一帧图片的旋转偏移矩阵赋值给meta

        # numpy.ma是针对掩码数组
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))     # 返回物体所对应的深度
        # print('mask_depth:', mask_depth)
        if self.mode == 'eval':                                         # 验证集中mask是单通道的，掩码中白色区域返回True，黑色返回False
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:                                                           # 标准数据中mask是3通道的，掩码中白色区域返回True，黑色返回False
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        # print('mask_label:', mask_label)
        mask = mask_label * mask_depth                                  # 把mask和深度图结合到一起，物体存在的区域像素为True

        if self.add_noise:
            img = self.trancolor(img)                                   # 对图像加入噪声
        img = np.array(img)[:, :, :3]                                   # [h, w, c] --> [c, h, w]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img
        if self.mode == 'eval':
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label)) # 是eval模式，根据mask_label获得目标的bbox
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])           # 不是eval模式，则从gt.yml中，获取最标准的bbox
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]                # 根据确定的行和列，对图像进行截取
        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))       # 获取目标物体旋转矩阵的参数
        target_t = np.array(meta['cam_t_m2c'])                          # 获取目标物体偏移矩阵参数

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]      # 将mask图片的目标区域裁剪，并拉成一维

        if len(choose) == 0:                                            # 如果面积为0，则代表没有目标物体
            cc = torch.LongTensor([0])
            return (cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num:                                      # 如果目标物体像素超过点云数目
            c_mask = np.zeros(len(choose), dtype=int)                   # 设定长度相同的c_mask
            c_mask[:self.num] = 1                                       # 前self.num设为1
            np.random.shuffle(c_mask)                                   # 打乱顺序
            choose = choose[c_mask.nonzero()]                           # 随机选取choose中的500个点
        else:                                                           # 如果目标物体像素小于点云数目
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')# 使用0填补将choose变成500
        # 得到choose挑选出来的500个索引

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)      # 把深度图里目标位置拉平并根据choose挑选坐标
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)   # 把原图里目标位置拉平并根据choose挑选坐标
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = 1.0                                                 # 从info.yml得知摄像头缩放参数均为1.0
        pt2 = depth_masked / cam_scale                                  # 深度图对应一维进行缩放
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy           # 对坐标进行标准化
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)                 # 把y, x, depth3个坐标合并在一起，变成点云数据
        print('cloud:', cloud)
        cloud = cloud / 1000.0                                          # 根据审读正则化
        print('cloud:', cloud)
        add_t = np.array([random.uniform(
            -self.noise_trans, self.noise_trans) for i in range(3)])    # 对偏移矩阵添加噪声
        if self.add_noise:                                              # 对点云添加噪声
            cloud = np.add(cloud, add_t)

        model_points = self.pt[obj] / 1000.0                            # 存储model/.ply中的点云数据，并将其进行正则化，就是目标物体的点云数据
        del_list = [j for j in range(0, len(model_points))]
        del_list = random.sample(del_list, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, del_list, axis=0)        # 删除多余的点云数据，只保留self.num_pt_mesh_small个

        target = np.dot(model_points, target_r.T)                       # 根据model_points第一帧目标模型对应的点云信息和target

        if self.add_noise:                                              # 目前迭代这张图片的旋转和偏移矩阵，计算出对应的点云数据
            target = np.add(target, target_t / 1000.0 + add_t)          # 根据是否添加噪声进行讨论
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0

        '''
        cloud: 由深度图计算出来的点云，该点云数据以本摄像头为参考坐标
        choose: 所选择点云的索引
        img_masked: 通过box剪切下来的RGB图像
        target: 根据model_points点云信息，以及旋转偏移矩阵转换过的点云信息
        model_points: 目标初始帧(模型)对应的点云信息
        self.objlist.index(obj): 目标物体的序列编号
        '''

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)])

    def __len__(self):                                                  # 常规重写
        return self.length

    def get_sym_list(self):                                             # 对称物体列表
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):                                      # 获取点云数目
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



# 将mask转换为box，把轮廓转化为矩形形状
def mask_to_bbox(mask):
    mask = mask.astype(np.unit8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    mask:寻找轮廓的图像，即传入的mask
    cv2.RETR_TREE:轮廓检索模式，利用等级树结构的轮廓
    cv2.CHAIN_APPROX_SIMPLE:水平方向，垂直方向，对角线方向，只保留终点坐标
    返回参数:
    contours: 轮廓本身
    '''
    x, y, w, h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)          # 使用最小矩形将轮廓包住，tmp_x, tmp_y为矩形左上角坐标, w,h为矩形的宽和高
        if tmp_w * tmp_h > w * h:                                       # 返回所有最小矩形中的最大的，能够包住所有轮廓
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]

# 与self.border_list相同
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

# 把bbox标准化，根据border_list中特定的大小
def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]      # [y, y + h, x, x + w]，方便之后进行边界处理
    # 边界处理
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin                                                   # bbox的高
    c_b = cmax - cmin                                                   # bbox的宽
    for i in range(len(border_list)):                                   # 设定标准的box，寻找轮廓高范围的最小box
        if r_b > border_list[i] and r_b <border_list[i + 1]:
            r_b = border_list[i + 1]
            break
    for i in range(len(border_list)):                                   # 设定标准的box，寻找轮廓宽范围的最小box
        if c_b > border_list[i] and c_b < border_list[i + 1]:
            c_b = border_list[i + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]           # 轮廓初始中心点，方便后面计算真实中心点
    rmin = center[0] - int(r_b / 2)                                     # 标准化后的轮廓最上侧
    rmax = center[0] + int(r_b / 2)                                     # 标准化后的轮廓最下侧
    cmin = center[0] - int(c_b / 2)                                     # 标准化后的轮廓最左侧
    cmax = center[0] + int(c_b / 2)                                     # 标准化后的轮廓最右侧
    #特殊值处理
    if rmin < 0:                                                        # 如果rmin超出边界,则整体向下移动-rmin个像素
        rmax = rmax + rmin
        rmin = 0
    if rmax > 480:
        rmin = rmin - (rmax - 480)                                      # 如果rmax超出边界，则整体向上移动(rmax-480)个像素
        rmax = 480
    if cmin < 0:                                                        # 如果cmin超出边界，则整体向右移动-cmin个像素
        cmax = cmax - cmin
        cmin = 0
    if cmax > 640:                                                      # 如果cmax超出边界，则整体向右移动(cmax-640)个像素
        cmin = cmin - (cmax - 640)
        cmax = 640
    return rmin, rmax, cmin, cmax


# 读取ply文件
def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == 'ply'
    f.readline()
    f.readline()                                                        # 略过开头两行
    N = int(f.readline().split()[-1])
    while f.readline().strip() != 'end_header':                         # 从end_header开始赋值给pts，前面是一些头标注
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)
