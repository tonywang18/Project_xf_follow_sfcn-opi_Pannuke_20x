#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Project_xf_follow_sfcn-opi_Pannuke_20x
@File    ：main.py.py
@IDE     ：PyCharm 
@Author  ：kiven
@Date    ：2022/3/15 15:18 
'''
import os
import torch
import torch.nn.functional as F
from DataReader.DataReader_Pannuke import DatasetReader
from config import project_root, device, net_in_hw, net_out_hw, batch_size, epochs, batch_counts,\
    is_train_from_recent_checkpoint, dataset_path, net_save_dir, train_lr,\
    match_distance_thresh_list, process_control, make_cla_is_det, use_repel_code,eval_save_path,data_path
import eval_utils
import uuid
import albumentations as albu
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import numpy as np
import loss_func
import copy
from wentai_toole import big_pic_result
import yaml
import cv2

get_pg_name = eval_utils.get_pg_name
get_pg_id = eval_utils.get_pg_id

in_dim = 3

cla_class_num = DatasetReader.class_num - 1               #   分类数
cla_class_ids = list(range(cla_class_num))                #   类别编号
cla_class_names = DatasetReader.class_names[:-1]          #   分类名字

save_dir = net_save_dir                                   # 模型存储路径

os.makedirs(save_dir, exist_ok=True)                      # 创建模型存储文件夹


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    image = batch[0]
    rough_mask = batch[1]
    fine_mask = batch[2]
    det_center_dict = batch[3]
    image_name = batch[4]
    del batch
    return image, rough_mask, fine_mask, det_center_dict, image_name

def tr_pm_to_onehot_vec(label_pm: torch.Tensor, soft_label=True):
    if soft_label:
        label_oh = label_pm
    else:
        label_oh = (label_pm > 0.5).type(torch.float32)

    bg = 1 - label_oh.max(1, keepdim=True)[0]
    label_oh = torch.cat([bg, label_oh], dim=1)
    return label_oh

def main(NetClass):
    model_id = NetClass.model_id
    single_id = str(uuid.uuid1())
    print('single_id:', single_id)

    #  ============================== 定义存储文件路径 =====================================

    ck_name = '{}/{}_model.pt'.format(save_dir, model_id)
    ck_best_name = '{}/{}_model_best.pt'.format(save_dir, model_id)

    ck_optim_name = '{}/{}_optim.pt'.format(save_dir, model_id)
    ck_restart_name = '{}/{}_restart.yml'.format(save_dir, model_id)

    score_det_name = '{}/{}_score_det.txt'.format(save_dir, model_id)
    score_cla_name = '{}/{}_score_cla.txt'.format(save_dir, model_id)

    score_det_best_name = '{}/{}_score_det_best.txt'.format(save_dir, model_id)
    score_cla_best_name = '{}/{}_score_cla_best.txt'.format(save_dir, model_id)



    # ===========================    定义数据增强的方式     =======================================
    transform = albu.Compose([     # 后面再仔细的调整
        albu.RandomCrop(64, 64, always_apply=True, p=1),
        albu.RandomRotate90(always_apply=False, p=0.5),   # 将输入随机旋转90度，零次或多次。
        albu.VerticalFlip(always_apply=False, p=0.5),
        albu.HorizontalFlip(always_apply=False, p=0.5),
        # albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
        #                  value=None, mask_value=None, always_apply=False, p=0.5), # 随机仿射变换

        # albu.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
        albu.HueSaturationValue(),
        albu.OneOf([
            # 模糊相关操作
            albu.MotionBlur(p=.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.OneOf([
            # 锐化、浮雕等操作
            albu.CLAHE(clip_limit=2),
            albu.Sharpen(),
            albu.Emboss(),
            albu.RandomBrightnessContrast(),
        ], p=0.3),
        albu.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度

    ])
    # 验证集不需要进行增强
    transform_valid = albu.Compose([
    ])

    train_dataset = DatasetReader(os.path.join(data_path), 'fold_1',
                                  pt_radius_min=9, pt_radius_max=9, transforms=transform,rescale=0.5,use_repel=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)

    valid_dataset = DatasetReader(os.path.join(data_path), 'fold_2',
                                  pt_radius_min=9, pt_radius_max=9, transforms=transform_valid,rescale=0.5,use_repel=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn=collate_fn)


    # 定义网络模型的结构，这四个数字分别是    1.模型的输入通道    2.粗检测输出通道    3. 细检测输出通道   4.分类输出通道
    net = NetClass()
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), train_lr, eps=1e-8, weight_decay=1e-6)
    optim_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, 1e-6)

    start_epoch = 0
    pg_max_valid_value = [-math.inf for _ in range(len(process_control)+1)]
    pg_minloss_value = [math.inf for _ in range(len(process_control)+1)]

    # ============================== 进行训练 阶段的切换  ==========================================
    if is_train_from_recent_checkpoint and os.path.isfile(ck_restart_name):
        d = yaml.safe_load(open(ck_restart_name, 'r'))
        start_epoch = d['start_epoch']
        pg_max_valid_value = d['pg_max_valid_value']
        pg_minloss_value = d['pg_minloss_value']

        if start_epoch not in process_control:
            new_ck_name = get_pg_name(ck_name, start_epoch, process_control)
            new_ck_optim_name = get_pg_name(ck_optim_name, start_epoch, process_control)
            net.load_state_dict(torch.load(new_ck_name, 'cpu'))
            optimizer.load_state_dict(torch.load(new_ck_optim_name, 'cpu'))

    net.train()  # 开启训练模式

    for epoch in range(start_epoch,epochs):
        optim_adjust.step(epoch)

        print("epoch",epoch)
        print("learning—rate",optimizer.state_dict()['param_groups'][0]['lr'])


        train_det_acc = 0
        train_cla_acc = 0
        train_loss = 0

        if epoch in process_control:     # 需要进行阶段的变更了
            new_ck_name = get_pg_name(ck_best_name,epoch-1,process_control)       # 获取上一个阶段最好的模型参数的路径
            net.load_state_dict(torch.load(new_ck_name,'cpu'))                # 进行模型参数的权重加载
            new_ck_optim_name = get_pg_name(ck_optim_name, epoch - 1, process_control)  # 获取到上一个epoch的优化器的参数
            optimizer.load_state_dict(torch.load(new_ck_optim_name, 'cpu'))  # 进行优化器参数的加载

        pg_id = get_pg_id(epoch, process_control)
        print("pg_id",pg_id)
        net.train()    # 开启训练模式



        if pg_id == 1:      # 完全复现文泰      粗检测阶段(关闭细检测和分类分支)
            net.enabled_b2_branch = False
            net.enabled_b3_branch = False
            net.set_freeze_seg1(False)
            net.set_freeze_seg2(True)
            net.set_freeze_seg3(True)
        elif pg_id == 2:                      #  细检测和分类阶段（冻结粗检测分支，同时训练细检测分支和分类分支）
            net.enabled_b2_branch = True
            net.enabled_b3_branch = True
            net.set_freeze_seg1(True)
            net.set_freeze_seg2(False)
            net.set_freeze_seg3(False)
        elif pg_id == 3:                       # 联合训练  同时训练粗检测，细检测，分类分支。
            net.enabled_b2_branch = True
            net.enabled_b3_branch = True
            net.set_freeze_seg1(False)
            net.set_freeze_seg2(False)
            net.set_freeze_seg3(False)
        else:
            raise AssertionError()

        for batch_count in tqdm(range(batch_counts)):
            for index,(image, rough_mask_final,fine_mask_final,label_center_dict,image_name) in enumerate(train_loader):

                # 将6个通道划分为5个通道和1个通道，最后一个通道是检测mask，但是不准确故舍弃
                mask_cla_label, _ = np.split(fine_mask_final, [5, ],-1)
                mask_det_label = np.max(mask_cla_label,  -1,keepdims=True)            # 细检测的mask repel_code

                image = torch.tensor(image, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255

                mask_det_label = torch.tensor(mask_det_label, dtype=torch.float32, device=device).permute(0, 3, 1,2)
                mask_cla_label = torch.tensor(mask_cla_label, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                mask_rough_label = torch.tensor(rough_mask_final, dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # 粗检测 0-1 mask


                batch_pred_det, batch_pred_det2, batch_pred_cla = net(image)

                # 训练时检查像素准确率，这里的准确率按照类别平均检查
                with torch.no_grad():
                    batch_pred_det_tmp = batch_pred_det
                    batch_pred_det_tmp = batch_pred_det_tmp.softmax(1)[:, 1:]
                    batch_pred_det_tmp_bin = (batch_pred_det_tmp > 0.5).type(torch.float32)
                    det_acc = (batch_pred_det_tmp_bin * mask_det_label).sum() / (mask_det_label.sum() + 1e-8).item()

                    if pg_id > 1:
                        batch_pred_cla_tmp = batch_pred_cla
                        batch_pred_cla_tmp = batch_pred_cla_tmp.clamp(0., 1.)

                        batch_pred_cla_tmp_bin = (batch_pred_cla_tmp > 0.5).type(torch.float32)
                        cla_acc = ((batch_pred_cla_tmp_bin * mask_cla_label).sum(dim=(0, 2, 3)) / (
                                    mask_cla_label.sum(dim=(0, 2, 3)) + 1e-8)).mean().item()

                        batch_pred_det2_tmp = batch_pred_det2.clamp(0., 1.)
                    else:
                        cla_acc = 0.


                if pg_id == 1:

                    loss = loss_func.det_loss(batch_pred_det.softmax(dim=1), mask_rough_label, 'auto')
                elif pg_id == 2:

                    _tmp = batch_pred_det.softmax(dim=1)[:, 1:2]
                    loss = loss_func.a_cla_loss_type3(_tmp, batch_pred_det2, mask_det_label, 'auto')
                    loss += loss_func.a_cla_loss_type3(_tmp, batch_pred_cla, mask_cla_label, 'auto')

                elif pg_id == 3:

                    loss1 = loss_func.det_loss(batch_pred_det.softmax(dim=1), mask_rough_label, 'auto')
                    _tmp = batch_pred_det.softmax(dim=1)[:, 1:2]
                    loss2 = loss_func.a_cla_loss_type3(_tmp, batch_pred_det2, mask_det_label, 'auto')
                    loss2 += loss_func.a_cla_loss_type3(_tmp, batch_pred_cla, mask_cla_label, 'auto')
                    loss = loss1 + loss2
                else:
                    raise AssertionError('Unknow process_step')

                train_det_acc += det_acc
                train_cla_acc += cla_acc
                train_loss += loss.item()


                print(
                    'epoch: {} count: {} train det acc: {:.3f} train cla acc: {:.3f} loss: {:.3f}'.format(epoch, batch_count, det_acc,
                                                                                                          cla_acc,
                                                                                                          loss.item()))
                optimizer.zero_grad()
                assert not np.isnan(loss.item()), 'Found loss Nan!'
                loss.backward()
                optimizer.step()

            train_det_acc = train_det_acc / (batch_count+1)
            train_cla_acc = train_cla_acc / (batch_count+1)
            train_loss = train_loss / (batch_count+1)


        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                net.eval()

                net.enabled_b2_branch = True
                net.enabled_b3_branch = True

                det_score_table = {'epoch': epoch,
                                   'det_pix_pred_found': 0,
                                   'det_pix_pred_fakefound': 0,
                                   'det_pix_label_found': 0,
                                   'det_pix_label_nofound': 0,
                                   'det_pix_recall': 0,
                                   'det_pix_prec': 0,
                                   'det_pix_f1': 0,
                                   'det_pix_f2': 0,
                                   }

                for dt in match_distance_thresh_list:
                    det_score_table[dt] = {
                        'found_pred': 0,  # 所有的假阳性
                        'fakefound_pred': 0,  # 所有的假阴性
                        'found_label': 0,  # 所有找到的标签
                        'nofound_label': 0,  # 所有找到的预测
                        'label_repeat': 0,  # 对应了多个pred的标签
                        'pred_repeat': 0,  # 对应了多个label的预测
                        'f1': None,
                        'recall': None,
                        'prec': None,
                    }

                cla_score_table = {'epoch': epoch}
                for dt in match_distance_thresh_list:
                    cla_score_table[dt] = {
                        'found_pred': 0,  # 所有的假阳性
                        'fakefound_pred': 0,  # 所有的假阴性
                        'found_label': 0,  # 所有找到的标签
                        'nofound_label': 0,  # 所有找到的预测
                        'label_repeat': 0,  # 对应了多个pred的标签
                        'pred_repeat': 0,  # 对应了多个label的预测
                        'f1': None,
                        'recall': None,
                        'prec': None,
                    }

                    for cls_id in cla_class_ids:
                        cla_score_table[dt][cls_id] = {
                            'found_pred': 0,  # 所有的假阳性
                            'fakefound_pred': 0,  # 所有的假阴性
                            'found_label': 0,  # 所有找到的标签
                            'nofound_label': 0,  # 所有找到的预测
                            'label_repeat': 0,  # 对应了多个pred的标签
                            'pred_repeat': 0,  # 对应了多个label的预测
                            'f1': None,
                            'recall': None,
                            'prec': None,
                        }

                for valid_index, (valid_image, valid_mask_rough_label, valid_mask_fine_label,valid_det_center_dict, valid_image_name) in enumerate(valid_loader):

# ================================== 不知道为什么这里获得的都是元组，所以为了取数据【0】 =======================
                    valid_det_center_dict = valid_det_center_dict[0]
                    valid_image = valid_image[0]
                    valid_image_name = valid_image_name[0]
                    valid_mask_rough_label = valid_mask_rough_label[0]
                    valid_mask_fine_label = valid_mask_fine_label[0]


# ==================================================================================================================
                    valid_mask_cla_label, _ = np.split(valid_mask_fine_label, [5, ], -1)
                    valid_mask_det_label = np.max(valid_mask_cla_label, -1, keepdims=True)

                    del valid_mask_fine_label


                    group_label_cla_pts = copy.deepcopy(valid_det_center_dict)  # 获取细胞核中心点坐标 存储的是5个通道的字典

                    label_det_pts = []
                    new_group_label_cla_pts = {}

                    for k in group_label_cla_pts:
                        # 排除det类，det类是不准确的
                        if k != 5:
                            label_det_pts.extend(group_label_cla_pts[k])
                            new_group_label_cla_pts[k] = np.array(group_label_cla_pts[k])

                    label_det_pts = np.array(label_det_pts)  # 存储的是细胞核的中心点坐标，用于检测，是一个数组


# ========================================= 开始进行大图的拼接预测 ===============================================
                    # 1 + 1 + cla_class_num 表示一共有几个通道，一个粗检测通道，一个细检测通道 5个分类通道
                    wim = big_pic_result.BigPicPatch(1+1+cla_class_num, [valid_image], (0, 0), net_in_hw, (0, 0), 0, 0,
                                                     custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255)
                    gen = wim.batch_get_im_patch_gen(batch_size*3)

                    for batch_info, batch_patch in gen:
                        batch_patch = torch.tensor(np.asarray(batch_patch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.
                        valid_batch_pred_det, valid_batch_pred_det2, valid_batch_pred_cla = net(batch_patch)

                        valid_batch_pred_det = valid_batch_pred_det.softmax(1)[:, 1:]   # 粗检测预测通道
                        valid_batch_pred_det2 = valid_batch_pred_det2.clamp(0, 1)       # 细检测预测通道
                        valid_batch_pred_cla = valid_batch_pred_cla.clamp(0, 1)         # 分类预测通道

                        valid_batch_pred = torch.cat([valid_batch_pred_det, valid_batch_pred_det2, valid_batch_pred_cla], 1)
                        valid_batch_pred = valid_batch_pred.permute(0, 2, 3, 1).cpu().numpy()
                        wim.batch_update_result(batch_info, valid_batch_pred)

                    pred_pm = wim.multi_scale_result[0].data / np.clip(wim.multi_scale_mask[0].data, 1e-8, None)
                    pred_det_pm, pred_det2_pm, pred_cla_pm = np.split(pred_pm, [1, 2], -1)

                    if pg_id > 1:
                        pred_det_final_pm = pred_det_pm * pred_det2_pm
                        pred_cla_final_pm = pred_det_pm * pred_cla_pm
                    else:
                        pred_det_final_pm = pred_cla_final_pm = False

                    if valid_image_name % 200 == 0:
                        out_dir = eval_save_path

# ================================================ mask路径 ====================================================================
                        eval_image_path = out_dir + str(epoch) + str(valid_image_name) + '_eval_image.png'
                        eval_mask_rough_det_path = out_dir + str(epoch) + str(valid_image_name) + '_eval_mask_rough_det.png'
                        eval_mask_fine_det_path = out_dir + str(epoch) + str(valid_image_name) + '_eval_mask_fine_det.png'

                        eval_mask_cla0_path = out_dir + str(epoch) + str(valid_image_name) + '_mask_cla_0.png'
                        eval_mask_cla1_path = out_dir + str(epoch) + str(valid_image_name) + '_mask_cla_1.png'
                        eval_mask_cla2_path = out_dir + str(epoch) + str(valid_image_name) + '_mask_cla_2.png'
                        eval_mask_cla3_path = out_dir + str(epoch) + str(valid_image_name) + '_mask_cla_3.png'
                        eval_mask_cla4_path = out_dir + str(epoch) + str(valid_image_name) + '_mask_cla_4.png'

                        # ======================================= 预测的路径 ========================================
                        pred_det_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_det.png'
                        pred_det_fine_before_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_det_fine_before.png'
                        pred_det_fine_after_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_det_fine_after.png'

                        pred_cla_0_before_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_0_before.png'
                        pred_cla_1_before_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_1_before.png'
                        pred_cla_2_before_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_2_before.png'
                        pred_cla_3_before_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_3_before.png'
                        pred_cla_4_before_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_4_before.png'

                        pred_cla_0_after_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_0_after.png'
                        pred_cla_1_after_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_1_after.png'
                        pred_cla_2_after_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_2_after.png'
                        pred_cla_3_after_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_3_after.png'
                        pred_cla_4_after_path = out_dir + str(epoch) + str(valid_image_name) + '_pred_cla_4_after.png'

                        cv2.imwrite(eval_image_path, valid_image)
                        cv2.imwrite(eval_mask_rough_det_path, valid_mask_rough_label * 255)
                        cv2.imwrite(eval_mask_fine_det_path, valid_mask_det_label * 255)

                        cv2.imwrite(eval_mask_cla0_path, (valid_mask_cla_label[:, :, 0]) * 255)
                        cv2.imwrite(eval_mask_cla1_path, (valid_mask_cla_label[:, :, 1]) * 255)
                        cv2.imwrite(eval_mask_cla2_path, (valid_mask_cla_label[:, :, 2]) * 255)
                        cv2.imwrite(eval_mask_cla3_path, (valid_mask_cla_label[:, :, 3]) * 255)
                        cv2.imwrite(eval_mask_cla4_path, (valid_mask_cla_label[:, :, 4]) * 255)

                        cv2.imwrite(pred_det_path, (pred_det_pm > 0.5) * 255)

                        if pg_id > 1:
                            cv2.imwrite(pred_det_fine_before_path, (pred_det2_pm > 0.5) * 255)
                            cv2.imwrite(pred_det_fine_after_path, (pred_det_final_pm > 0.5) * 255)

                            cv2.imwrite(pred_cla_0_before_path, (pred_cla_pm[:, :, 0] > 0.5) * 255)
                            cv2.imwrite(pred_cla_1_before_path, (pred_cla_pm[:, :, 1] > 0.5) * 255)
                            cv2.imwrite(pred_cla_2_before_path, (pred_cla_pm[:, :, 2] > 0.5) * 255)
                            cv2.imwrite(pred_cla_3_before_path, (pred_cla_pm[:, :, 3] > 0.5) * 255)
                            cv2.imwrite(pred_cla_4_before_path, (pred_cla_pm[:, :, 4] > 0.5) * 255)

                            cv2.imwrite(pred_cla_0_after_path, (pred_cla_final_pm[:, :, 0] > 0.5) * 255)
                            cv2.imwrite(pred_cla_1_after_path, (pred_cla_final_pm[:, :, 1] > 0.5) * 255)
                            cv2.imwrite(pred_cla_2_after_path, (pred_cla_final_pm[:, :, 2] > 0.5) * 255)
                            cv2.imwrite(pred_cla_3_after_path, (pred_cla_final_pm[:, :, 3] > 0.5) * 255)
                            cv2.imwrite(pred_cla_4_after_path, (pred_cla_final_pm[:, :, 4] > 0.5) * 255)

                    # 统计像素分类数据
                    if pg_id == 1:
                        det_pix_pred_bin = (pred_det_pm[..., 0:1] > 0.5).astype(dtype=np.float32)
                    else:
                        det_pix_pred_bin = (pred_det_final_pm[..., 0:1] > 0.5).astype(dtype=np.float32)

                    det_pix_label_bin = (valid_mask_det_label[..., 0:1] > 0.5).astype(dtype=np.float32)


                    det_score_table['det_pix_pred_found'] += float((det_pix_pred_bin * det_pix_label_bin).sum(dtype=np.float32))
                    det_score_table['det_pix_pred_fakefound'] += float((det_pix_pred_bin * (1 - det_pix_label_bin)).sum(dtype=np.float32))
                    det_score_table['det_pix_label_found'] += det_score_table['det_pix_pred_found']
                    det_score_table['det_pix_label_nofound'] += float(((1 - det_pix_pred_bin) * det_pix_label_bin).sum(dtype=np.float32))



# ====================================================   获得检测的匹配结果    ==============================================
                    # 统计点找到数据
                    if pg_id == 1:
                        _tmp_pred_det_pm = pred_det_pm
                    else:
                        _tmp_pred_det_pm = pred_det_final_pm

                    det_info = eval_utils.calc_a_sample_info_points_each_class(_tmp_pred_det_pm, [label_det_pts,[0] * len(label_det_pts)],
                                                                               [0], match_distance_thresh_list,
                                                                               use_post_pro=False,
                                                                               use_single_pair=True)



# ====================================================   获得分类的匹配结果    ==============================================

                    tmp_label_cla_without_ignore_pos = []
                    tmp_label_cla_without_ignore_cla = []
                    for k in group_label_cla_pts:
                        tmp_label_cla_without_ignore_pos.extend(group_label_cla_pts[k])     # 细胞核的中心坐标点 没有分类位置
                        tmp_label_cla_without_ignore_cla.extend([k] * len(group_label_cla_pts[k]))   # 带有分类位置的 细胞核中心坐标点

                    if pg_id > 1:
                        cla_info = eval_utils.calc_a_sample_info_points_each_class(pred_cla_final_pm,
                                                                                   [tmp_label_cla_without_ignore_pos,tmp_label_cla_without_ignore_cla],cla_class_ids,
                                                                                   match_distance_thresh_list,
                                                                                   use_post_pro=False,
                                                                                   use_single_pair=True)
                    else:
                        cla_info = None


                    for dt in match_distance_thresh_list:
                        for cls in [0]:
                            det_score_table[dt]['found_pred'] += det_info[cls][dt]['found_pred']
                            det_score_table[dt]['fakefound_pred'] += det_info[cls][dt]['fakefound_pred']
                            det_score_table[dt]['found_label'] += det_info[cls][dt]['found_label']
                            det_score_table[dt]['nofound_label'] += det_info[cls][dt]['nofound_label']
                            det_score_table[dt]['pred_repeat'] += det_info[cls][dt]['pred_repeat']
                            det_score_table[dt]['label_repeat'] += det_info[cls][dt]['label_repeat']

                    if cla_info is not None:
                        for dt in match_distance_thresh_list:
                            for cls in cla_class_ids:
                                cla_score_table[dt][cls]['found_pred'] += cla_info[cls][dt]['found_pred']
                                cla_score_table[dt][cls]['fakefound_pred'] += cla_info[cls][dt]['fakefound_pred']
                                cla_score_table[dt][cls]['found_label'] += cla_info[cls][dt]['found_label']
                                cla_score_table[dt][cls]['nofound_label'] += cla_info[cls][dt]['nofound_label']
                                cla_score_table[dt][cls]['pred_repeat'] += cla_info[cls][dt]['pred_repeat']
                                cla_score_table[dt][cls]['label_repeat'] += cla_info[cls][dt]['label_repeat']

                                cla_score_table[dt]['found_pred'] += cla_info[cls][dt]['found_pred']
                                cla_score_table[dt]['fakefound_pred'] += cla_info[cls][dt]['fakefound_pred']
                                cla_score_table[dt]['found_label'] += cla_info[cls][dt]['found_label']
                                cla_score_table[dt]['nofound_label'] += cla_info[cls][dt]['nofound_label']
                                cla_score_table[dt]['pred_repeat'] += cla_info[cls][dt]['pred_repeat']
                                cla_score_table[dt]['label_repeat'] += cla_info[cls][dt]['label_repeat']


                # 计算det的像素的F1，精确率，召回率
                det_score_table['det_pix_prec'] = det_score_table['det_pix_pred_found'] / (det_score_table['det_pix_pred_fakefound'] + det_score_table['det_pix_pred_found'] + 1e-8)
                det_score_table['det_pix_recall'] = det_score_table['det_pix_label_found'] / (det_score_table['det_pix_label_nofound'] + det_score_table['det_pix_label_found'] + 1e-8)
                det_score_table['det_pix_f1'] = 2 * det_score_table['det_pix_prec'] * det_score_table['det_pix_recall'] / (det_score_table['det_pix_prec'] + det_score_table['det_pix_recall'] + 1e-8)
                det_score_table['det_pix_f2'] = 5 * det_score_table['det_pix_prec'] * det_score_table['det_pix_recall'] / (det_score_table['det_pix_prec'] * 4 + det_score_table['det_pix_recall'] + 1e-8)

                # 计算det的F1，精确率，召回率
                for dt in match_distance_thresh_list:
                    prec = det_score_table[dt]['found_pred'] / (det_score_table[dt]['found_pred'] + det_score_table[dt]['fakefound_pred'] + 1e-8)
                    recall = det_score_table[dt]['found_label'] / (det_score_table[dt]['found_label'] + det_score_table[dt]['nofound_label'] + 1e-8)
                    f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                    det_score_table[dt]['prec'] = prec
                    det_score_table[dt]['recall'] = recall
                    det_score_table[dt]['f1'] = f1

                # 计算cla的F1，精确率，召回率
                for dt in match_distance_thresh_list:
                    prec = cla_score_table[dt]['found_pred'] / (cla_score_table[dt]['found_pred'] + cla_score_table[dt]['fakefound_pred'] + 1e-8)
                    recall = cla_score_table[dt]['found_label'] / (cla_score_table[dt]['found_label'] + cla_score_table[dt]['nofound_label'] + 1e-8)
                    f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                    cla_score_table[dt]['prec'] = prec
                    cla_score_table[dt]['recall'] = recall
                    cla_score_table[dt]['f1'] = f1

                    for cls_id in cla_class_ids:
                        prec = cla_score_table[dt][cls_id]['found_pred'] / (
                                cla_score_table[dt][cls_id]['found_pred'] + cla_score_table[dt][cls_id][
                            'fakefound_pred'] + 1e-8)
                        recall = cla_score_table[dt][cls_id]['found_label'] / (
                                cla_score_table[dt][cls_id]['found_label'] + cla_score_table[dt][cls_id][
                            'nofound_label'] + 1e-8)
                        f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                        cla_score_table[dt][cls_id]['prec'] = prec
                        cla_score_table[dt][cls_id]['recall'] = recall
                        cla_score_table[dt][cls_id]['f1'] = f1

                # 打印评分
                out_line = yaml.safe_dump(det_score_table) + '\n' + yaml.safe_dump(cla_score_table)
                print('epoch {}'.format(epoch), out_line)

                if pg_id == 1:
                    # current_valid_value = det_score_table[match_distance_thresh_list[0]]['f1']
                    current_valid_value = det_score_table['det_pix_f2']
                else:
                    current_valid_value = cla_score_table[match_distance_thresh_list[0]]['f1'] + \
                                          det_score_table[match_distance_thresh_list[0]]['f1']

                # 保存最好的
                if current_valid_value > pg_max_valid_value[pg_id - 1]:
                    pg_max_valid_value[pg_id - 1] = current_valid_value
                    new_ck_best_name = get_pg_name(ck_best_name, epoch, process_control)
                    torch.save(net.state_dict(), new_ck_best_name)
                    # new_ck_best_extra_name = get_pg_name(ck_best_extra_name, e)
                    # open(new_ck_best_extra_name, 'w').write(out_line)
                    new_score_det_best_name = get_pg_name(score_det_best_name, epoch, process_control)
                    new_score_cla_best_name = get_pg_name(score_cla_best_name, epoch, process_control)
                    yaml.safe_dump(det_score_table, open(new_score_det_best_name, 'w'))
                    yaml.safe_dump(cla_score_table, open(new_score_cla_best_name, 'w'))

                new_ck_name = get_pg_name(ck_name, epoch, process_control)
                new_ck_optim_name = get_pg_name(ck_optim_name, epoch, process_control)
                torch.save(net.state_dict(), new_ck_name)
                torch.save(optimizer.state_dict(), new_ck_optim_name)
                d = {'start_epoch': epoch + 1, 'pg_max_valid_value': pg_max_valid_value,
                     'pg_minloss_value': pg_minloss_value}
                yaml.safe_dump(d, open(ck_restart_name, 'w'))
                new_score_det_name = get_pg_name(score_det_name, epoch, process_control)
                new_score_cla_name = get_pg_name(score_cla_name, epoch, process_control)
                yaml.safe_dump(det_score_table, open(new_score_det_name, 'w'))
                yaml.safe_dump(cla_score_table, open(new_score_cla_name, 'w'))



if __name__ == '__main__':
    from model.model_1_v1 import MainNet
    main(MainNet)



