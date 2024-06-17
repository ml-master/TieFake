from torchvision import models
from torch import nn
import numpy as np
import torch
import os
import sys
from resnest.torch import resnest101
import warnings  # 导入警告模块
warnings.filterwarnings("ignore")  # 忽略所有警告

def vgg19_2way(pretrained: bool):
    model_ft = models.vgg19(pretrained=pretrained)
    model_ft.classifier.add_module("fc",nn.Linear(1000,1))
    return model_ft

def resnet50_2way(pretrained: bool):
    model_ft = models.resnet50(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    return model_ft

# def resnest101_2way(pretrained:bool):
#     model_ft = resnest101(pretrained=pretrained)
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Linear(num_ftrs, 1)
#     return model_ft

# resnest101.py
def resnest101_2way(pretrained=False, local_weight_path=None):
    # 如果提供了本地权重路径，禁用在线预训练模型加载
    if pretrained and local_weight_path:
        model_ft = resnest101(pretrained=False)
        state_dict = torch.load(local_weight_path)
        model_ft.load_state_dict(state_dict)
    else:
        # 直接使用在线预训练模型（如果pretrained为True）
        model_ft = resnest101(pretrained=pretrained)

    # 替换最后一层，使其适合二分类任务
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)  # 输出层有1个神经元
    
    # 其他模型配置代码
    return model_ft


if __name__ == "__main__":
    model_ft = resnest101_2way(pretrained=True)
