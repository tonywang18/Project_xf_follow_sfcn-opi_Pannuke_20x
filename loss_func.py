#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Project_xf_follow_sfcn-opi_Pannuke 
@File    ：loss_func.py
@IDE     ：PyCharm 
@Author  ：kiven
@Date    ：2022/3/16 18:35 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union


# 权重固定
def det_loss(batch_pred_det:torch.Tensor,batch_label_det,det_weights:Union[torch.Tensor,str]='auto'):
    '''
    注意，这里要求 y_true 是 onehot向量，而不是类别标量
    :param batch_pred_det:
    :param batch_label_cla:
    :param det_weights:
    :return:
    '''
    batch_label_det_final = torch.cat([1-batch_label_det, batch_label_det], dim=1)   # 通道0背景，通道1是细胞核预测
    pos_n = batch_label_det_final[:, 1].sum()
    neg_n = batch_label_det_final[:, 0].sum()
    a = neg_n / pos_n
    det_weights = torch.tensor([1, a], dtype=torch.float32, device=batch_label_det.device)
    det_weights = torch.reshape(det_weights, [1, -1, 1, 1])

    batch_pred_det = torch.where(batch_pred_det > 1e-10, batch_pred_det, batch_pred_det + 1e-10)
    loss = -((batch_label_det_final * torch.log(batch_pred_det) * det_weights )).sum() /  (batch_pred_det.shape[0] * batch_pred_det.shape[1] * batch_pred_det.shape[2] * batch_pred_det.shape[3])

    return loss



def a_cla_loss_type3(batch_pred_det, batch_pred_cla: torch.Tensor, batch_label_cla: torch.Tensor, cla_weights: Union[torch.Tensor, str]='auto', threshold=0.5):
    '''
    注意，这里要求 batch_label_cla 是 onehot 向量，而不是类别标量
    :param batch_pred_det:
    :param batch_pred_cla:
    :param batch_label_cla:
    :param cla_weights:
    :param threshold:
    :return:
    '''
    indicator = (batch_pred_det.detach() >= threshold).type(torch.float32)

    cla_weights = torch.ones(1, batch_pred_cla.shape[1], 1, 1, dtype=torch.float32, device=batch_pred_det.device)
    if isinstance(cla_weights, str) and cla_weights == 'auto' and batch_pred_cla.shape[1] != 1:
        # 方法1，只使用权重1
        # cla_weights = torch.tensor([1]*batch_label_cla.shape[1], dtype=torch.float32, device=batch_label_cla.device)
        # cla_weights = cla_weights.reshape(1, -1, 1, 1)
        # 方法2，类别平衡
        batch_label_cla_masked = batch_label_cla * indicator
        each_label_cla_pix_num = batch_label_cla_masked.sum(dim=(0, 2, 3), keepdim=True)
        most_cla = torch.max(each_label_cla_pix_num, 1, keepdim=True)[0]
        each_label_cla_pix_num = torch.clamp_min(each_label_cla_pix_num, 1)
        cla_weights = most_cla / each_label_cla_pix_num
        cla_weights = torch.log(cla_weights + 1) / 0.113328685307003

    A = 1
    loss = ((1 + batch_label_cla * A) * cla_weights * (batch_label_cla - batch_pred_cla).abs()).pow(2)
    loss = (loss * indicator).sum() / indicator.sum()

    return loss
