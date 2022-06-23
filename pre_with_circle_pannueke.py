import os
import torch
import torch.nn.functional as F
import time
import numpy as np
import yaml
import imageio
import copy
from DataReader.DataReader_Pannuke import DatasetReader
from config import project_root, device, net_in_hw, net_out_hw, batch_size, epochs, batch_counts,\
    is_train_from_recent_checkpoint, dataset_path, net_save_dir, train_lr,\
    match_distance_thresh_list, process_control, make_cla_is_det, use_repel_code,eval_which_checkpoint,eval_save_path,data_path
import albumentations as albu
from wentai_toole import big_pic_result
from torch.utils.data import DataLoader
import eval_utils
from my_py_lib import list_tool
from tqdm import tqdm
from eval_utils import calc_a_sample_info_points_each_class, class_map_to_contours2
import cv2


# =======================================  一些配置性变量数据 =====================================
use_heatmap_nms = True
use_single_pair = True

pred_heatmap_thresh_list = (0.6,0.5,0.4,0.3,0.2,0.1)


cla_class_num = DatasetReader.class_num - 1
cla_class_ids = list(range(cla_class_num))
cla_class_names = DatasetReader.class_names[:-1]

pred_train_out_dir = 'pred_circle/cla_pred_circle_train_out_dir'
pred_valid_out_dir = 'pred_circle/cla_pred_circle_valid_out_dir'
pred_test_out_dir = 'pred_circle/cla_pred_circle_test_out_dir'


get_pg_id = eval_utils.get_pg_id
get_pg_name = eval_utils.get_pg_name


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



def heatmap_nms(hm: np.ndarray):
    a = (hm * 255).astype(np.int32)
    a1 = cv2.blur(hm, (3, 3)).astype(np.int32)
    a2 = cv2.blur(hm, (5, 5)).astype(np.int32)
    a3 = cv2.blur(hm, (7, 7)).astype(np.int32)

    ohb = (hm > 0.).astype(np.float32)

    h = a + a1 + a2 + a3

    h = (h / 4).astype(np.float32)

    ht = torch.tensor(h)[None, None, ...]
    htm = torch.nn.functional.max_pool2d(ht, 9, stride=1, padding=4)
    hmax = htm[0, 0, ...].numpy()

    h = (h >= hmax).astype(np.float32)

    h = h * ohb

    h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return h


def main(NetClass):
    torch.set_grad_enabled(False)

    model_id = NetClass.model_id

# ======================= 获取权重路径 =================================
    ck_name         = '{}/{}_model.pt'          .format(net_save_dir, model_id)
    ck_best_name    = '{}/{}_model_best.pt'     .format(net_save_dir, model_id)

    start_epoch = 299
    pg_id = get_pg_id(start_epoch, process_control)

    print("pg_id", pg_id)

    # 测试时不需要进行增强
    transform_valid = albu.Compose([
    ])


    train_dataset = DatasetReader(os.path.join(data_path), 'fold_1',
                                  pt_radius_min=1, pt_radius_max=1, transforms=transform_valid,rescale=0.5,use_repel=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)

    valid_dataset = DatasetReader(os.path.join(data_path), 'fold_2',
                                  pt_radius_min=1, pt_radius_max=1, transforms=transform_valid,rescale=0.5,use_repel=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn=collate_fn)

    test_dataset = DatasetReader(os.path.join(data_path), 'fold_3',
                                  pt_radius_min=1, pt_radius_max=1, transforms=transform_valid,rescale=0.5,use_repel=False)
    test_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn=collate_fn)






# ===================  定义网络模型 =========================
    net = NetClass()
    net = net.to(device)
    net.enabled_b2_branch = True
    net.enabled_b3_branch = True

    if eval_which_checkpoint == 'last':
        print('Will load last weight.')
        new_ck_name = get_pg_name(ck_name, start_epoch, process_control)
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))
        print('load model success')
    elif eval_which_checkpoint == 'best':
        print('Will load best weight.')
        new_ck_name = get_pg_name(ck_best_name, start_epoch, process_control)
        print("new_ck_name", new_ck_name)
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))

        print('load model success')

    else:
        print('Unknow weight type. Will not load weight.')

    net = net.to(device)
    net.eval()

    # 进行预测
    for heatmap_thresh in pred_heatmap_thresh_list:
        # for did, cur_dataset in enumerate([train_dataset, valid_dataset, test_dataset]):
        #     out_dir = [pred_train_out_dir, pred_valid_out_dir, pred_test_out_dir][did]
        for did, cur_dataset in enumerate([test_dataset]):
            out_dir = [pred_test_out_dir][did]

            out_dir = '{}/{}'.format(out_dir, str(heatmap_thresh))  # 预测结果保存路径

            os.makedirs(out_dir, exist_ok=True)  # 创建文件夹

            det_score_table = {}
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

            det_score_table_a2 = copy.deepcopy(det_score_table)  # 假阳性抑制分支的评分

            cla_score_table = {}
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
# =============================  选择 测试的数据集 =============================
            if cur_dataset == train_dataset:
                cur_dataloader = train_loader
            elif cur_dataset == valid_dataset:
                cur_dataloader = valid_loader
            elif cur_dataset == test_dataset:
                cur_dataloader = test_loader

            for valid_index, (valid_image, valid_mask_rough_label, valid_mask_fine_label,valid_det_center_dict, valid_image_name) in enumerate(
                    valid_loader):
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

# ==================================  标签mask，细胞中心坐标 ====================
                label_det_pts = []  # 存储细胞中心坐标，没有分类信息，用于检测
                group_label_cls_pts = {}  # 存储细胞中心坐标，有分类信息
                for k, v in valid_det_center_dict.items():
                    # 类别5是不准确的，需要去除
                    if k == 5:
                        continue
                    label_det_pts.extend(v)
                    if k not in group_label_cls_pts:
                        group_label_cls_pts[k] = []
                    group_label_cls_pts[k].append(v)

                label_det_pts = np.array(label_det_pts)
                for c in group_label_cls_pts:
                    group_label_cls_pts[c] = np.reshape(group_label_cls_pts[c], [-1, 2])

                print('Processing {}'.format(valid_image_name))  # 正在预测的图片编号

                del valid_det_center_dict

                # ========================================= 开始进行大图的拼接预测 ===============================================
                # 1 + 1 + cla_class_num 表示一共有几个通道，一个粗检测通道，一个细检测通道 5个分类通道
                wim = big_pic_result.BigPicPatch(1 + 1 + cla_class_num, [valid_image], (0, 0), net_in_hw, (0, 0), 0, 0,
                                                 custom_patch_merge_pipe=eval_utils.patch_merge_func,
                                                 patch_border_pad_value=255)
                gen = wim.batch_get_im_patch_gen(batch_size * 3)

                for batch_info, batch_patch in gen:
                    batch_patch = torch.tensor(np.asarray(batch_patch), dtype=torch.float32, device=device).permute(0,3,1,2) / 255.
                    valid_batch_pred_det, valid_batch_pred_det2, valid_batch_pred_cla = net(batch_patch)

                    valid_batch_pred_det = valid_batch_pred_det.softmax(1)[:, 1:]  # 粗检测预测通道
                    valid_batch_pred_det2 = valid_batch_pred_det2.clamp(0, 1)  # 细检测预测通道
                    valid_batch_pred_cla = valid_batch_pred_cla.clamp(0, 1)  # 分类预测通道

                    valid_batch_pred = torch.cat([valid_batch_pred_det, valid_batch_pred_det2, valid_batch_pred_cla], 1)
                    valid_batch_pred = valid_batch_pred.permute(0, 2, 3, 1).cpu().numpy()
                    wim.batch_update_result(batch_info, valid_batch_pred)

                pred_pm = wim.multi_scale_result[0].data / np.clip(wim.multi_scale_mask[0].data, 1e-8, None)
                pred_det_pm, pred_det2_pm, pred_cla_pm = np.split(pred_pm, [1, 2], -1)

                pred_det_final_pm = pred_det_pm * pred_det2_pm
                pred_cla_final_pm = pred_det_pm * pred_cla_pm

# =========================================  粗检测  ===============================================

                # 获得相应的预测细胞核中心的坐标点
                pred_det_rough_pts = eval_utils.get_pts_from_hm(pred_det_pm, heatmap_thresh)

                # 在图片上面画上预测的点和mask标注点
                mix_pic_det_a1 = eval_utils.draw_hm_circle(valid_image, pred_det_rough_pts, label_det_pts, 6)

                # 进行评分
                det_info = calc_a_sample_info_points_each_class([pred_det_rough_pts, [0] * len(pred_det_rough_pts)],
                                                                [label_det_pts, [0] * len(label_det_pts)], [0],
                                                                match_distance_thresh_list,
                                                                use_post_pro=False, use_single_pair=use_single_pair)

                for dt in match_distance_thresh_list:
                    for cls in [0]:
                        det_score_table[dt]['found_pred'] += det_info[cls][dt]['found_pred']
                        det_score_table[dt]['fakefound_pred'] += det_info[cls][dt]['fakefound_pred']
                        det_score_table[dt]['found_label'] += det_info[cls][dt]['found_label']
                        det_score_table[dt]['nofound_label'] += det_info[cls][dt]['nofound_label']
                        det_score_table[dt]['pred_repeat'] += det_info[cls][dt]['pred_repeat']
                        det_score_table[dt]['label_repeat'] += det_info[cls][dt]['label_repeat']

                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a1_image.png'.format(valid_image_name)), mix_pic_det_a1)
                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a1_PM.png'.format(valid_image_name)),
                                (pred_det_pm * 255).astype(np.uint8))
                yaml.dump(det_info, open(os.path.join(out_dir, '{}_det.txt'.format(valid_image_name)), 'w'))



# ==============================================  细检测  ==================================================
                if use_heatmap_nms:
                    pred_det_final_pm[..., 0] = pred_det_final_pm[..., 0] * heatmap_nms(pred_det_final_pm[..., 0])

                # 从细检测热图获得预测细胞核中心的坐标点
                pred_det_post_pts = eval_utils.get_pts_from_hm(pred_det_final_pm, heatmap_thresh)

                # 画图
                mix_pic_det_a2 = eval_utils.draw_hm_circle(valid_image, pred_det_post_pts, label_det_pts, 6)

                # 评分
                det_info_a2 = calc_a_sample_info_points_each_class(
                    [pred_det_post_pts, [0] * len(pred_det_post_pts)],
                    [label_det_pts, [0] * len(label_det_pts)], [0],
                    match_distance_thresh_list, use_single_pair=use_single_pair)

                for dt in match_distance_thresh_list:
                    for cls in [0]:
                        det_score_table_a2[dt]['found_pred'] += det_info_a2[cls][dt]['found_pred']
                        det_score_table_a2[dt]['fakefound_pred'] += det_info_a2[cls][dt]['fakefound_pred']
                        det_score_table_a2[dt]['found_label'] += det_info_a2[cls][dt]['found_label']
                        det_score_table_a2[dt]['nofound_label'] += det_info_a2[cls][dt]['nofound_label']
                        det_score_table_a2[dt]['pred_repeat'] += det_info_a2[cls][dt]['pred_repeat']
                        det_score_table_a2[dt]['label_repeat'] += det_info_a2[cls][dt]['label_repeat']

                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a2_image.png'.format(valid_image_name)), mix_pic_det_a2)
                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a2_h_1before_PM.png'.format(valid_image_name)),
                                (pred_det_pm * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(out_dir, '{}_1_det_a2_h_2after_PM.png'.format(valid_image_name)),
                                (pred_det_final_pm * 255).astype(np.uint8))
                yaml.dump(det_info_a2, open(os.path.join(out_dir, '{}_det_a2.txt'.format(valid_image_name)), 'w'))



# =================================================  分类检测 =======================================
                pred_cla_final_pm_hm_before = np.copy(pred_cla_final_pm)  # 分类 进行nms操作之前的热图

                # 进行nms操作      对分类的每个通道都进行nsm操作
                if use_heatmap_nms:
                    for c in range(pred_cla_final_pm.shape[2]):
                        pred_cla_final_pm[:, :, c] = pred_cla_final_pm[:, :, c] * heatmap_nms(
                            pred_cla_final_pm[:, :, c])

                ## 新方法，从det中获得   预测的相关
                pred_cla_pts = []  # 获得检测mask的细胞核的中心点坐标
                pred_cla_pts_cla = []  # 这里面存储的是检测的细胞核中心坐标点所对应的类别号
                pred_cla_pts.extend(pred_det_post_pts)  # 从细检测结果获取坐标点
                pred_cla_pts_cla.extend(eval_utils.get_cls_pts_from_hm(pred_cla_pts, pred_cla_final_pm_hm_before))

                # 标签的相关
                label_cla_pts = []  # 标签 细胞核中心点坐标
                label_cla_pts_cla = []  # 标签 细胞核中心点坐标 分类
                for i in group_label_cls_pts:
                    label_cla_pts.extend(group_label_cls_pts[i])
                    label_cla_pts_cla.extend([i] * len(group_label_cls_pts[i]))

                cla_info = calc_a_sample_info_points_each_class([pred_cla_pts, pred_cla_pts_cla],
                                                                [label_cla_pts, label_cla_pts_cla], cla_class_ids,
                                                                match_distance_thresh_list,
                                                                use_single_pair=use_single_pair)

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

                group_mix_pic_cla = []  # 保存的是图片，保存的是测试的图片，预测和标签的中心点画在了上面
                for i in range(cla_class_num):
                    cur_pred_pts = list_tool.list_multi_get_with_bool(pred_cla_pts,
                                                                      np.array(pred_cla_pts_cla, np.int32) == i)
                    cur_label_pts = list_tool.list_multi_get_with_bool(label_cla_pts,
                                                                       np.array(label_cla_pts_cla, np.int32) == i)
                    group_mix_pic_cla.append(eval_utils.draw_hm_circle(valid_image, cur_pred_pts, cur_label_pts, 6))

                # 输出分类分支的图
                for i in range(cla_class_num):
                    # 图片样子
                    imageio.imwrite(os.path.join(out_dir, '{}_2cla_b1_{}.png'.format(valid_image_name, cla_class_names[i])),
                                    group_mix_pic_cla[i])

                    # 生成分类分支的处理前和处理后的热图
                    imageio.imwrite(
                        os.path.join(out_dir, '{}_2cla_b2_{}_1before.png'.format(valid_image_name, cla_class_names[i])),
                        (pred_cla_pm[..., i] * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(out_dir, '{}_2cla_b2_{}_2after.png'.format(valid_image_name, cla_class_names[i])),
                        (pred_cla_final_pm[..., i] * 255).astype(np.uint8))



                # 计算det a1 F1，精确率，召回率
            for dt in match_distance_thresh_list:
                prec = det_score_table[dt]['found_pred'] / (
                        det_score_table[dt]['found_pred'] + det_score_table[dt]['fakefound_pred'] + 1e-8)
                recall = det_score_table[dt]['found_label'] / (
                        det_score_table[dt]['found_label'] + det_score_table[dt]['nofound_label'] + 1e-8)
                f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                det_score_table[dt]['prec'] = prec
                det_score_table[dt]['recall'] = recall
                det_score_table[dt]['f1'] = f1

            yaml.dump(det_score_table, open(os.path.join(out_dir, 'all_det.txt'), 'w'))

            # 计算det a2 F1，精确率，召回率
            for dt in match_distance_thresh_list:
                prec = det_score_table_a2[dt]['found_pred'] / (
                            det_score_table_a2[dt]['found_pred'] + det_score_table_a2[dt]['fakefound_pred'] + 1e-8)
                recall = det_score_table_a2[dt]['found_label'] / (
                            det_score_table_a2[dt]['found_label'] + det_score_table_a2[dt]['nofound_label'] + 1e-8)
                f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                det_score_table_a2[dt]['prec'] = prec
                det_score_table_a2[dt]['recall'] = recall
                det_score_table_a2[dt]['f1'] = f1

            yaml.dump(det_score_table_a2, open(os.path.join(out_dir, 'all_det_a2.txt'), 'w'))

            # 计算cla F1，精确率，召回率
            for dt in match_distance_thresh_list:
                prec = cla_score_table[dt]['found_pred'] / (
                            cla_score_table[dt]['found_pred'] + cla_score_table[dt]['fakefound_pred'] + 1e-8)
                recall = cla_score_table[dt]['found_label'] / (
                            cla_score_table[dt]['found_label'] + cla_score_table[dt]['nofound_label'] + 1e-8)
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

            yaml.dump(cla_score_table, open(os.path.join(out_dir, 'all_cla.txt'), 'w'))

if __name__ == '__main__':
    from model.model_1_v1 import MainNet
    main(MainNet)