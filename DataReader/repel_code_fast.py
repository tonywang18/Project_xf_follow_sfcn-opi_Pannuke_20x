#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ProjectD_xf_follow_wentai 
@File    ：repel_code_fast.py
@IDE     ：PyCharm 
@Author  ：kiven
@Date    ：2022/3/2 15:50 
'''
import numpy as np
from skimage.draw import disk as sk_disk
def draw_repel_code_fast(im: np.ndarray, pts: np.ndarray, r, A=1):
    '''
    Another faster acceleration method code of the above function, but there are accuracy problems.
    :param im: [H, W, 1] The image to be drawn.
    :param pts: [N, yx] Input points. You need to ensure that the each point is unique.
    :param r: The radius of the drawing area.
    :param A: Control the gradual change.
    :return:
    '''
    assert im.ndim == 3 and im.shape[2] == 1

    for pt in pts:
        ds = np.linalg.norm(pt[None,] - pts, 2, 1)
        # 注意，这个点集已经包含了待查询点自身
        # 该处r*2是为了增加更多的检查点，减少不准确问题
        close_pts = pts[ds <= r * 2]
        rr, cc = sk_disk(pt, r, shape=im.shape[:2])
        pts2 = np.stack([rr, cc], 1)
        pts2 = np.asarray(pts2, np.int32)

        big_ds = np.linalg.norm(np.asarray(pts2[:, None] - close_pts[None], np.float32), 2, 2)
        # big_ds = np.linalg.norm(np.asarray(close_pts[None,] - pts2[:, None], np.float32), 2, 2)
        big_seq_id = np.argsort(big_ds, axis=-1)[:, :2]
        big_close_2pts_ds = np.take_along_axis(big_ds, big_seq_id, 1)

        if big_close_2pts_ds.shape[1] == 2:
            D = big_close_2pts_ds[:, 0] * (1 + big_close_2pts_ds[:, 0] / (np.clip(big_close_2pts_ds[:, 1], 1e-8, None)))
            C = np.where(D < r, 1 / (1 + D * A), 0)
        elif big_close_2pts_ds.shape[1] == 1:
            D = big_close_2pts_ds[:, 0]
            C = np.where(D < r, 1 / (1 + D * A), 0)
        else:
            # return im
            continue
        im[pts2[:, 0], pts2[:, 1]] = C[:, None]

    return im
