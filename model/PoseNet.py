#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from model.pspnet import PSPNet
import torch.nn.functional as F
from torchsummary import summary
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18')
}


# 之前的pspnet模型，用于语义分割
class ModifiedResnet(nn.Module):
    def __init__(self):
        super(ModifiedResnet, self).__init__()
        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        return self.model(x)


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        '''

        :param num_points: 点云数目
        '''
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        '''

        :param x: geometry embeddings[bs, 3, 500]
        :param emb: color embeddings[bs, 32, 500]
        :return:
        '''

        x = F.relu(self.conv1(x))                                               # geo_emb:[bs, 3, 500]->[bs, 64, 500]
        emb = F.relu(self.e_conv1(emb))                                         # col_emb:[bs, 32, 500]->[bs, 64, 500]
        pointfeat_1 = torch.cat((x, emb), dim=1)                                # [geo_emb, col_emb] = [bs, 128, 500]
        x = F.relu(self.conv2(x))                                               # geo_emb:[bs, 64, 500]->[bs, 128, 500]
        emb = F.relu(self.e_conv2(emb))                                         # col_emb:[bs, 64, 500]->[bs, 128, 500]
        pointfeat_2 = torch.cat((x, emb), dim=1)                                # [geo_emb, col_emb] = [bs, 256, 500]
        x = F.relu(self.conv5(pointfeat_2))                                     # [bs, 512, 500]
        x = F.relu(self.conv6(x))                                               # [bs, 1024, 500]
        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        print('ap_x----:',ap_x.shape)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)                   # [bs, 128 + 256 + 1024,500]


class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        '''

        :param num_points: 点云数目
        :param num_obj: 目标物体种类
        '''
        super(PoseNet, self).__init__()                                         # 继承Module的八大基本参数
        self.num_points = num_points                                            # 点云的数目
        self.cnn = ModifiedResnet()                                             # resenet模型
        # summary(self.cnn, (3, 120, 120))
        self.feat = PoseNetFeat(num_points)
        # summary(self.feat, [(3, 500), (32, 500)])
        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)                            # 依次对应旋转、偏移、置信
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj * 4, 1)
        self.conv4_t = torch.nn.Conv1d(128, num_obj * 3, 1)
        self.conv4_c = torch.nn.Conv1d(128, num_obj * 1, 1)

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        '''

        :param img: RGB图像[bs, 3, h, w]
        :param x: 点云数据[bs, 500, 3]
        :param choose: 所选择的点云下标[bs, 1, 500]
        :param obj: 目标所对应的物体序列号[bs, 1]
        :return:图像的姿态预测
        '''

        out_img = self.cnn(img)                                                 # out_img: [bs, 3, h, w]->[bs, 32, h, w]
        bs, di, _, _ = out_img.size()                                           # 提取bs和维度, bs = bs; di = 32
        emb = out_img.view(bs, di, -1)                                          # resize
        choose = choose.repeat(1, di, 1)                                        # choose: [bs, 1, 500]->[bs, 32, 500]
        emb = torch.gather(emb, 2, choose).contiguous()                         # emb: [bs, 32, -1]->[bs, 32, 500]
        x = x.transpose(2, 1).contiguous()                                      # x: [bs, 500, 3]->[bs, 3, 500]
        print('!!!!!-------------!!!!!!', x.shape, emb.shape)
        ap_x = self.feat(x, emb)                                                # ap_x: [bs, 1408, 500]

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)        # rx: [bs, num_obj, 4, num_points]
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)        # tx: [bs, num_obj, 3, num_points]
        cx = torch.sigmoid(self.conv4_c(cx).view(bs, self.num_obj, 1, self.num_points))     # cx: [bs, num_obj, 1, num_points]
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])                           # out_rx: [bs, 4, 500]
        out_tx = torch.index_select(tx[b], 0, obj[b])                           # out_tx: [bs, 3, 500]
        out_cx = torch.index_select(cx[b], 0, obj[b])                           # out_cx: [bs, 1, 500]
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()               # out_rx: [bs, 500, 4]
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()               # out_tx: [bs, 500, 3]
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()               # out_cx: [bs, 500, 1]
        return out_rx, out_tx, out_cx, emb.detach()



