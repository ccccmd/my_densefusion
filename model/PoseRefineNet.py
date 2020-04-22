#-*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)

        self.e_conv1 = nn.Conv1d(32, 64, 1)
        self.e_conv2 = nn.Conv1d(64, 128, 1)

        self.conv5 = nn.Conv1d(384, 512, 1)
        self.conv6 = nn.Conv1d(512, 1024, 1)

        self.ap1 = nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):

        x = F.relu(self.conv1(x))                       # 点云第一次卷积
        emb = F.relu(self.e_conv1(emb))                 # 图像第一次卷积
        pointfeat_1 = torch.cat([x, emb], dim=1)        # 点云图像第一次融合

        x = F.relu(self.conv2(x))                       # 点云第二次卷积
        emb = F.relu(self.e_conv2(emb))                 # 图像第二次卷积
        pointfeat_2 = torch.cat([x, emb], dim=1)        # 点云图像第二次融合

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)  # 64 + 64 + 128 +128 = 384

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))
        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)

        return ap_x


class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = nn.Linear(1024, 512)
        self.conv1_t = nn.Linear(1024, 512)

        self.conv2_r = nn.Linear(512, 128)
        self.conv2_t = nn.Linear(512, 128)

        self.conv3_r = nn.Linear(128, num_obj * 4)  # 旋转
        self.conv3_t = nn.Linear(128, num_obj * 3)  # 偏移

    def forward(self, x, emb, obj):
        bs = x.size()[0]                            # batchsize

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)                    # PoseRefineNetFeat

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx



