#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Project_xf_follow_sfcn-opi_Pannuke_20x
@File    ：DataReader_Pannuke.py
@IDE     ：PyCharm 
@Author  ：kiven
@Date    ：2022/3/15 15:19 
'''
import os
import pickle
from DataReader.repel_code_fast import draw_repel_code_fast
from torch.utils.data import DataLoader,Dataset
import albumentations as albu
import gzip
import cv2
import skimage.io
import torch
import numpy as np
import copy
from my_py_lib import contour_tool
from skimage.transform import rotate as sk_rotate
from skimage.draw import disk as sk_disk
import random

project_root = os.path.split(__file__)[0]
print(project_root)


class DatasetReader:
    class_names = ('Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial', 'Det')  # 分类的名字，det是背景
    class_ids = (0, 1, 2, 3, 4, 5)
    class_num = len(class_ids)
    assert len(class_names) == len(class_ids)
    def __init__(self, dataset_path=os.path.join('F:\生物图腾公司的数据集和代码\pannuke_packed_dataset'), data_type='fold_1', pt_radius_min=3, pt_radius_max=3,transforms=None,use_repel=False,rescale=1):
        data_type = '{}.pkl.gz'.format(data_type)                       # 获得数据集是哪个文件夹，以此来判断是训练集还是验证集
        self.data_path = os.path.join(dataset_path, data_type)          # 获取得到数据集的位置
        self.pt_radius_min = pt_radius_min              # 标签mask的圆的半径大小
        self.pt_radius_max = pt_radius_max
        self.transform = transforms
        self.use_repel = use_repel
        self.rescale = rescale
        assert self.pt_radius_max >= self.pt_radius_min


        self._encode_im_list = []      # 存储 图片数据
        self._im_type_list = []        # 存储 图片 组织类型
        self._label_list = []          # 存储 标签 数据

        self._cla_label_id = dict([(i, []) for i in self.class_ids])

# ==========================  保存最终的结果 ======================================
        self.image_results = []
        self.mask_fine_label_results = []             # 细检测采用的是repel_code里面是6个通道
        self.mask_rough_label_results = []            # 粗检测采用的是0 1 编码，一个通道
        self.label_center_dict = []
        self.image_name_results = []


        self._build()

    def _build(self):

        all_data = pickle.loads(gzip.decompress(open(self.data_path, 'rb').read()))
        self._encode_im_list.extend(all_data['encode_ims'])
        self._im_type_list.extend(all_data['im_types'])
        self._label_list.extend(all_data['labels'])

        for i, label_cls_conts in enumerate(self._label_list):
            for c in self.class_ids:
                if len(label_cls_conts[c]) > 0:
                    self._cla_label_id[c].append(i)

        #    最终检查
        assert len(self._encode_im_list) == len(self._im_type_list) == len(self._label_list)

        for id in range(len(self._encode_im_list)):  # 迭代每一张图片以及对应的label
            encode_im = self._encode_im_list[id]
            im_type = self._im_type_list[id]
            label_cls_conts = self._label_list[id]


            im = cv2.imdecode(encode_im[1], -1)                 # 获取到图片数据
            assert im.dtype == np.uint8

            if self.rescale != 1.:   # 对图片进行下采样
                im = cv2.resize(im, (int(im.shape[0] * self.rescale), int(im.shape[1] * self.rescale)),interpolation=cv2.INTER_AREA)



            label_cls_center_pts = copy.deepcopy(label_cls_conts)  # 对标签mask进行深拷贝

            for k in label_cls_center_pts.keys():
                for i in range(len(label_cls_center_pts[k])):
                    cont = label_cls_center_pts[k][i]
                    if self.rescale != 1.:
                        cont = contour_tool.resize_contours([cont], self.rescale)[0]
                    box = contour_tool.make_bbox_from_contour(cont)
                    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.int32)
                    label_cls_center_pts[k][i] = center


# ======================================= 画出粗检测的mask 这里的mask用的是sfcn-opi里面的画法 ==========================
            rough_label = np.zeros([im.shape[0], im.shape[1], 1], np.float32)
            for key in label_cls_center_pts.keys():
                if key == 5:
                    continue
                for (x, y) in label_cls_center_pts[key]:
                    x = np.round(x).astype(np.int32)
                    y = np.round(y).astype(np.int32)
                    cv2.circle(rough_label, center=(y, x), radius=3, color=1, thickness=-1)

# ===============  根据标签mask的坐标中心，画出细检测mask  ================================================

            fine_label = np.zeros([im.shape[0], im.shape[1], self.class_num], np.float32)

            for k in label_cls_center_pts.keys():
                    pts = label_cls_center_pts[k]
                    if len(pts) == 0:
                        continue
                    pts = np.array(pts, np.int32)
                    if not self.use_repel:
                        for pt in pts:
                            radius = random.uniform(self.pt_radius_min,self.pt_radius_max) if self.pt_radius_max > self.pt_radius_min else self.pt_radius_min
                            radius = int(np.round(radius))
                            rr, cc = sk_disk(pt, radius, shape=fine_label.shape[:2])
                            fine_label[rr, cc, k] = 1
                    else:
                        draw_repel_code_fast(fine_label[:, :, k:k + 1], pts, self.pt_radius_max, A=0.5)
            label_center = copy.deepcopy(label_cls_center_pts)



            self.image_results.append(im)
            self.mask_fine_label_results.append(fine_label)
            self.mask_rough_label_results.append(rough_label)
            self.label_center_dict.append(label_center)
            self.image_name_results.append(id)




    def __getitem__(self, item):  # 返回数据和标签
        image = self.image_results[item]
        rough_mask = self.mask_rough_label_results[item]
        fine_mask = self.mask_fine_label_results[item]
        label_center_dict = self.label_center_dict[item]
        image_name = self.image_name_results[item]

        #  将粗检测和细检测的mask进行堆叠进行统一的数据增强
        mask_det_all = np.stack((rough_mask[:, :, 0], fine_mask[:, :, 0], fine_mask[:, :, 1], fine_mask[:, :, 2],
                                 fine_mask[:, :, 3], fine_mask[:, :, 4],fine_mask[:, :, 5]), axis=2)

        img_mask_enhance = self.transform(image=image, mask=mask_det_all)
        image_final = img_mask_enhance['image']                                      # 获得数据增强后的 image
        mask_label = img_mask_enhance['mask']                                # 获得数据增强后的 mask标签
        rough_mask_final,fine_mask_final  = np.split(mask_label, [1], -1)


        return image_final, rough_mask_final,fine_mask_final,label_center_dict,image_name




    def __len__(self):
        return len(self._encode_im_list)



if __name__ == '__main__':
    def collate_fn(batch):
        #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
        batch = list(zip(*batch))
        image = batch[0]
        rough_mask = batch[1]
        fine_mask = batch[2]
        det_center_dict = batch[3]
        image_name = batch[4]
        del batch
        return image, rough_mask,fine_mask, det_center_dict, image_name

    transform = albu.Compose([
        # albu.RandomCrop(64, 64)  # 进行图片的变形
    ])  # 定义数据增强方式
    train_dataset = DatasetReader(r'F:\生物图腾公司的数据集和代码\pannuke_packed_dataset\\','fold_1',pt_radius_min=9, pt_radius_max=9,transforms=transform,use_repel=True,rescale=0.5)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn)
    for epoch in range(100):
        for index, (image, rough_mask_final,fine_mask_final,label_center_dict,image_name) in enumerate(train_loader):
            image = image[0][:, :, :]
            rough_mask_final_0 = rough_mask_final[0][:, :, :]
            fine_mask_final_0 = fine_mask_final[0][:, :, :]
            det_center_dict = label_center_dict[0]

            image_name = image_name[0]
            cv2.imshow('image', image)
            cv2.imshow('mask_cla_0', fine_mask_final_0[:, :, 0])
            cv2.imshow('mask_rough', rough_mask_final_0[:, :, 0])

            cv2.waitKey(0)
