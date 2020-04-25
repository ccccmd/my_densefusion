#-*-coding:utf-8-*-
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', type=str, default='')         # 验证数据根目录
parser.add_argument('--model', type=str, default='')                # PoseNet网络模型
parser.add_argument('--refine_model', type=str, default='')         # PoseRefineNet网络模型
opt = parser.parse_args()

