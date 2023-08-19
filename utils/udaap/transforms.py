from __future__ import absolute_import

import numpy as np
import torch
import skimage

from .misc import to_torch
from .imutils import im_to_torch, im_to_numpy


def color_normalize(x, mean, std):
    if x.size(0) == 1:  # 黑白图处理
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):  # 彩色图处理
        t.sub_(m)  # 去均值，未对方差进行处理。
    return x


def flip_back(flip_output, dataset):
    """
    flip output map
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    elif dataset == '_300w':
        matchedParts = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                        [17, 26], [18, 25], [19, 26], [20, 23], [21, 22], [36, 45], [37, 44],
                        [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                        [49, 53], [48, 54], [61, 63], [62, 64], [67, 65], [59, 55], [58, 56])
    elif dataset == 'scut':
        matchedParts = ([1, 21], [2, 20], [3, 19], [4, 18], [5, 17], [6, 16], [7, 15],
                        [8, 14], [9, 13], [10, 12], [26, 32], [25, 33], [24, 34], [23, 35],
                        [22, 36], [27, 41], [28, 40], [29, 39], [30, 38], [31, 37],
                        [49, 55], [48, 56], [47, 57], [46, 50], [45, 51], [44, 52], [43, 53], [42, 54], [58, 59],
                        [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 67],
                        [79, 73], [78, 74], [77, 75], [80, 85], [81, 84], [82, 83])
    elif dataset == 'real_animal':
        matchedParts = ([0, 1], [3, 4], [5, 6], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17])
    else:
        print('Not supported dataset: ' + dataset)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    # 这是本文的一个特有定义，所有关键点之间都是1vs1连接，所以他的左右翻转，就是这个1vs1连接的左右翻转。
    # 在更通用场景下的水平翻转并不适用。
    for pair in matchedParts:  # 关键点pair对的对称变换。
        tmp = np.copy(flip_output[:, pair[0], :, :])  # 取pair对中第1个关键点的预测结果。
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp  # 以上。前后交换

    return torch.from_numpy(flip_output).float()


# 对kps的水平翻转。（只使用其对坐标修改的部分，不用其kp-pair的翻转）
def shufflelr_ori(x, width, dataset, isMatchedParts=True):
    """
    flip coords
    """

    # 对坐标值的修改（Flip horizontal）
    x[:, 0] = width - x[:, 0]

    if isMatchedParts:
        # 定义各数据集中的kp-pair（这部分弃用，不适用）
        if dataset == 'mpii':
            matchedParts = (
                [0,5],   [1,4],   [2,3],
                [10,15], [11,14], [12,13]
            )
        elif dataset == '_300w':
            matchedParts = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                            [17, 26], [18, 25], [19, 26], [20, 23], [21, 22], [36, 45], [37, 44],
                            [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                            [49, 53], [48, 54], [61, 63], [62, 64], [67, 65], [59, 55], [58, 56])
        elif dataset == 'scut':
            matchedParts = ([1, 21], [2, 20], [3, 19], [4, 18], [5, 17], [6, 16], [7, 15],
                            [8, 14], [9, 13], [10, 12], [26, 32], [25, 33], [24, 34], [23, 35],
                            [22, 36], [27, 41], [28, 40], [29, 39], [30, 38], [31, 37],
                            [49, 55], [48, 56], [47, 57], [46, 50], [45, 51], [44, 52], [43, 53], [42, 54], [58, 59],
                            [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 67],
                            [79, 73], [78, 74], [77, 75], [80, 85], [81, 84], [82, 83])
        elif dataset == 'real_animal':
            matchedParts = ([0, 1], [3, 4], [5, 6], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17])
        else:
            print('Not supported dataset: ' + dataset)

        # 对kp-pair的翻转（这部分弃用，不适用）
        # 这是本文的一个特有定义，所有关键点之间都是1vs1连接，所以他的左右翻转，就是这个1vs1连接的左右翻转。
        # 在更通用场景下的水平翻转并不适用。
        for pair in matchedParts:  # 关键点pair对的对称变换。
            tmp = x[pair[0], :].clone()  # 取pair对中第1个关键点的预测结果。
            x[pair[0], :] = x[pair[1], :]
            x[pair[1], :] = tmp  # 以上。前后交换
    return x


# 对图像的水平翻转。
def fliplr(x):
    if x.ndim == 3:
        # np.fliplr 左右翻转
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def flip_weights(w):
    flipparts = [1, 0, 2, 4, 3, 6, 5, 7, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]
    return w[flipparts]


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # 生成转换矩阵（Generate transformation matrix），稍后详细调研。
    ## 计算平移矩阵。
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    ## 计算旋转矩阵。
    if not rot == 0:
        rot = -rot  # 匹配裁剪后的旋转方向（To match direction of rotation from cropping）
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]  # (0, 0): cos, (1, 0): -sin
        rot_mat[1, :2] = [sn, cs]  # (0, 0): sin, (1, 0): cos
        rot_mat[2, 2] = 1  # (2, 2): 1，其余为0。
        # 绕中心旋转（Need to rotate around center）
        t_mat = np.eye(3)  # 生成对角矩阵。
        t_mat[0, 2] = -res[1]/2  # (0, 2): -1*width/2
        t_mat[1, 2] = -res[0]/2  # (1, 2): -1*height/2
        t_inv = t_mat.copy()  # 以上2点+对角线都为1.0
        t_inv[:2, 2] *= -1  # 以上2个乘-1的地方，再把符号改回来。
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))  # ((对角矩阵*平移矩阵)*旋转矩阵)*取反的对角矩阵。
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # 计算放射变换矩阵（Transform pixel location to different reference）
    t = get_transform(center, scale, res, rot=rot)
    if invert:  # 是否求逆矩阵
        t = np.linalg.inv(t)  # 矩阵求逆矩阵。设A是一个n阶矩阵，若存在另一个n阶矩阵B，使得： AB=BA=E ，则称方阵A可逆，并称方阵B是A的逆矩阵。其中，E为单位矩阵。
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


# 对预测结果进行仿射变换（对关键点的预测坐标值进行处理）
def transform_preds(coords, center, scale, res):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, res, 1, 0))
    return coords

# 图像裁剪
def crop_ori(img, center, scale, res, rot=0):
    img = im_to_numpy(img)  # CxHxW ==> H*W*C

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]  # sf是什么？？？
    if sf < 2:
        sf = 1  # 小于2的，取整为1。
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))  # 取图像最长边的缩放后长度为最新尺寸，int(maxLength/sf)
        new_ht = int(np.math.floor(ht / sf))  # 计算缩放后的height
        new_wd = int(np.math.floor(wd / sf))  # 计算缩放后的width
        if new_size < 2:  # 图像过小（最长边小于2pixels的），设置为256*256*3的0矩阵（h*w*c）。
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            # img = scipy.misc.imresize(img, [new_ht, new_wd])
            img = skimage.transform.resize(img, (new_ht, new_wd))  # 依据计算后的height、width，resize图像。
            center = center * 1.0 / sf  # 重新计算中心点
            scale = scale / sf  # 重新计算scale

    # 计算左上角(0, 0)点转换后的点坐标（Upper left point）
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # 计算右下角(res, res)点转换后的点坐标（Bottom right point）
    br = np.array(transform(res, center, scale, res, invert=1))

    # 填充，当旋转时，适当数量的上下文被包括在内（Padding so that when rotated proper amount of context is included）
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)  # 与范式相关。稍后调研。
    if not rot == 0:  # 不旋转时不用padding，旋转时为保证kps不会出图像范围，则添加padding。
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]  # 添加RGB维度，生成height*3
    new_img = np.zeros(new_shape)

    # 要填充新数组的范围（Range to fill new array）
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # 从原始图像到样本的范围（Range to sample from original image）
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        # new_img = scipy.misc.imrotate(new_img, rot)
        new_img = skimage.transform.rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    # new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    new_img = im_to_torch(skimage.transform.resize(new_img, tuple(res)))
    return new_img
