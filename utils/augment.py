# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
import torch
import skimage
import torch.nn.functional as F

from .process import ProcessUtils as proc
from .udaap.imutils import im_to_numpy, im_to_torch
from .udaap.transforms import transform


class AugmentUtils:
    def __init__(self):
        pass

    @classmethod
    def affine(cls, imgMap, kpsMap, center, scale, sf, angle, rf, matrixRes):
        scale = scale * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
        angle = angle + torch.randn(1).mul_(rf).clamp(- rf, rf)[0] if random.random() <= 1.0 else 0.
        imgMap = cls.affine_image(imgMap, center, scale, matrixRes, angle)
        kpsMap = cls.affine_kps(kpsMap, center, scale, matrixRes, angle)
        return imgMap, kpsMap, scale, angle

    @classmethod
    def affine_mulKps(cls, imgMap, kpsMapArray, center, scale, sf, angle, rf, matrixRes):
        scale = scale * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
        angle = angle + torch.randn(1).mul_(rf).clamp(- rf, rf)[0] if random.random() <= 1.0 else 0.
        imgMap = cls.affine_image(imgMap, center, scale, matrixRes, angle)
        kpsMapArray_new = []
        for kpsMap in kpsMapArray:
            kpsMapArray_new.append(cls.affine_kps(kpsMap, center, scale, matrixRes, angle))
        return imgMap, kpsMapArray_new, scale, angle  # kpsMapArray_new ==> Stack. 待修改 202303

    @classmethod
    def affine_back2(cls, heatmap, warpmat, isflip):
        heatmap_back = heatmap.clone()
        # 进行反向仿射变换
        affine_grid = F.affine_grid(warpmat, heatmap_back.size(), align_corners=True)
        heatmap_back = F.grid_sample(heatmap_back, affine_grid, align_corners=True)
        # 进行反向水平翻转
        heatmaps_f = []
        for hIdx in range(len(heatmap_back)):
            heatmap_f = cls.fliplr_back_tensor(heatmap_back[hIdx]) if isflip[hIdx] else heatmap_back[hIdx]
            heatmaps_f.append(heatmap_f)
        return torch.stack(heatmaps_f, dim=0)

    @classmethod
    def affine_classification(cls, imgMap, center, scale, sf, angle, rf, matrixRes):
        scale = scale * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
        angle = angle + torch.randn(1).mul_(rf).clamp(- rf, rf)[0] if random.random() <= 1.0 else 0.
        imgMap = cls.affine_image(imgMap, center, scale, matrixRes, angle)
        return imgMap, scale, angle

    @classmethod
    def affine_mulKps_classification(cls, imgMap, center, scale, sf, angle, rf, matrixRes):
        scale = scale * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
        angle = angle + torch.randn(1).mul_(rf).clamp(- rf, rf)[0] if random.random() <= 1.0 else 0.
        imgMap = cls.affine_image(imgMap, center, scale, matrixRes, angle)
        return imgMap, scale, angle  # kpsMapArray_new ==> Stack. 待修改 202303

    @classmethod
    def affine_back2_classification(cls, heatmap, warpmat, isflip):
        heatmap_back = heatmap.clone()
        # 进行反向仿射变换
        affine_grid = F.affine_grid(warpmat, heatmap_back.size(), align_corners=True)
        heatmap_back = F.grid_sample(heatmap_back, affine_grid, align_corners=True)
        # 进行反向水平翻转
        heatmaps_f = []
        for hIdx in range(len(heatmap_back)):
            heatmap_f = cls.fliplr_back_tensor(heatmap_back[hIdx]) if isflip[hIdx] else heatmap_back[hIdx]
            heatmaps_f.append(heatmap_f)
        return torch.stack(heatmaps_f, dim=0)

    # region function：图像仿射变换（含随机缩放、随机旋转）
    # params：
    #   （1）imgMap：imgMap对象（C*H*W）
    #   （2）center：中心点
    #   （3）scale：缩放比例
    #   （4）matrixRes：变换矩阵尺寸（一般指输入网络的图像尺寸）
    #   （5）angle：旋转角度
    # return：
    #   （1）imgMap：操作后的imgMap对象（C*H*W）
    # endregion
    @classmethod
    def affine_image(cls, imgMap, center, scale, matrixRes, angle=0):
        image = im_to_numpy(imgMap)  # CxHxW (3, 256, 256) ==> H*W*C (256, 256, 3)
        # Preprocessing for efficient cropping
        ht, wd = image.shape[0], image.shape[1]
        sf = scale * 200.0 / matrixRes[0]  # sf是什么？？？
        if sf < 2:
            sf = 1  # 小于2的，取整为1。
        else:
            new_size = int(np.math.floor(max(ht, wd) / sf))  # 取图像最长边的缩放后长度为最新尺寸，int(maxLength/sf)
            new_ht = int(np.math.floor(ht / sf))  # 计算缩放后的height
            new_wd = int(np.math.floor(wd / sf))  # 计算缩放后的width
            if new_size < 2:  # 图像过小（最长边小于2pixels的），设置为256*256*3的0矩阵（h*w*c）。
                return torch.zeros(matrixRes[0], matrixRes[1], image.shape[2]) \
                    if len(image.shape) > 2 else torch.zeros(matrixRes[0], matrixRes[1])
            else:
                # img = scipy.misc.imresize(img, [new_ht, new_wd])
                image = skimage.transform.resize(image, (new_ht, new_wd))  # 依据计算后的height、width，resize图像。
                center = center * 1.0 / sf  # 重新计算中心点
                scale = scale / sf  # 重新计算scale

        # 计算左上角(0, 0)点转换后的点坐标（Upper left point）
        ul = np.array(transform([0, 0], center, scale, matrixRes, invert=1))
        # 计算右下角(res, res)点转换后的点坐标（Bottom right point）
        br = np.array(transform(matrixRes, center, scale, matrixRes, invert=1))

        # 填充，当旋转时，适当数量的上下文被包括在内（Padding so that when rotated proper amount of context is included）
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)  # 与范式相关。稍后调研。
        if not angle == 0:  # 不旋转时不用padding，旋转时为保证kps不会出图像范围，则添加padding。
            ul -= pad
            br += pad

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(image.shape) > 2:
            new_shape += [image.shape[2]]  # 添加RGB维度，生成height*3
        new_img = np.zeros(new_shape)

        # 要填充新数组的范围（Range to fill new array）
        new_x = max(0, -ul[0]), min(br[0], image.shape[1]) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], image.shape[0]) - ul[1]
        # 从原始图像到样本的范围（Range to sample from original image）
        old_x = max(0, ul[0]), min(image.shape[1], br[0])
        old_y = max(0, ul[1]), min(image.shape[0], br[1])
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = image[old_y[0]:old_y[1], old_x[0]:old_x[1]]

        if not angle == 0:
            # Remove padding
            # new_img = scipy.misc.imrotate(new_img, rot)
            new_img = skimage.transform.rotate(new_img, angle)
            new_img = new_img[pad:-pad, pad:-pad]
        # new_img = im_to_torch(scipy.misc.imresize(new_img, res))
        new_img = im_to_torch(skimage.transform.resize(new_img, tuple(matrixRes)))
        return new_img

    # region function：关键点仿射变换（含随机缩放、随机旋转） -- 一般与affine_image对应使用
    # params：
    #   （1）kpsMap：关键点集合（tensor对象集合）
    #   （2）center：中心点
    #   （3）scale：缩放比例
    #   （4）matrixRes：变换矩阵尺寸（一般指输入网络的图像尺寸）
    #   （5）angle：旋转角度
    # return：
    #   （1）kpsMap：操作后的关键点集合
    # endregion
    @classmethod
    def affine_kps(cls, kpsMap, center, scale, matrixRes, angle=0):
        kpsMap_affined = kpsMap.clone()
        for kIdx, kp in enumerate(kpsMap):
            if kpsMap[kIdx, 1] > 0:  # 坐标中的y值大于0（既该点可见）
                kpsMap_affined[kIdx, 0:2] = torch.from_numpy(transform(kpsMap[kIdx, 0:2], center, scale, matrixRes, rot=angle))
        return kpsMap_affined

    @classmethod
    def affine_getWarpmat(cls, angle, scale, matrixRes=[64, 64]):
        # 根据旋转和比例生成变换矩阵
        M = cv2.getRotationMatrix2D((int(matrixRes[0]/2), int(matrixRes[1]/2)), angle, 1 / scale)
        warpmat = cv2.invertAffineTransform(M)
        warpmat[:, 2] = 0
        return torch.Tensor(warpmat)

    @classmethod
    def affine_kpsFromHeatmap(cls, kps, center, scale, heatmapRes):
        for p in range(kps.size(0)):
            kps[p, 0:2] = torch.from_numpy(transform(kps[p, 0:2], center, scale, heatmapRes, 1, 0))
        return kps

    # region function：随机水平翻转
    # params：
    #   （1）imgMap：imgMap对象（C*H*W）
    #   （2）kps：keypoint集合
    #   （3）center：中心点
    #   （4）prob：触发概率
    # return：
    #   （1）imgMap：操作后的imgMap对象（C*H*W）
    #   （2）kps：操作后的keypoint集合
    #   （3）center：操作后的中心点
    #   （4）isflip：是否触发翻转
    # endregion
    @classmethod
    def fliplr(cls, imgMap, kpsMap, center, prob=0.5):
        isflip = False
        if random.random() <= prob:
            imgMap = torch.from_numpy(proc.image_fliplr(imgMap.numpy())).float()
            kpsMap = proc.kps_fliplr(kpsMap, imgMap.size(2))
            center[0] = imgMap.size(2) - center[0]
            isflip = True
        return imgMap, kpsMap, center, torch.tensor(isflip)

    @classmethod
    def fliplr_classification(cls, imgMap, center, prob=0.5):
        isflip = False
        if random.random() <= prob:
            imgMap = torch.from_numpy(proc.image_fliplr(imgMap.numpy())).float()
            center[0] = imgMap.size(2) - center[0]
            isflip = True
        return imgMap, center, torch.tensor(isflip)

    # region function：随机水平翻转
    # params：
    #   （1）imgMap：imgMap对象（C*H*W）
    #   （2）kps：keypoint集合
    #   （3）center：中心点
    #   （4）prob：触发概率
    # return：
    #   （1）imgMap：操作后的imgMap对象（C*H*W）
    #   （2）kps：操作后的keypoint集合
    #   （3）center：操作后的中心点
    #   （4）isflip：是否触发翻转
    # endregion
    @classmethod
    def fliplr_mulKps(cls, imgMap, kpsMapArray, center, prob=0.5):
        isflip = False
        if random.random() <= prob:
            imgMap = torch.from_numpy(proc.image_fliplr(imgMap.numpy())).float()
            kpsMapArray_new = []
            for kpsMap in kpsMapArray:
                kpsMapArray_new.append(proc.kps_fliplr(kpsMap, imgMap.size(2)))
            center[0] = imgMap.size(2) - center[0]
            isflip = True
        else:
            kpsMapArray_new = kpsMapArray
        return imgMap, kpsMapArray_new, center, torch.tensor(isflip)

    @classmethod
    def fliplr_mulKps_classification(cls, imgMap, center, prob=0.5):
        isflip = False
        if random.random() <= prob:
            imgMap = torch.from_numpy(proc.image_fliplr(imgMap.numpy())).float()
            center[0] = imgMap.size(2) - center[0]
            isflip = True
        return imgMap, center, torch.tensor(isflip)

    @classmethod
    def fliplr_back(cls, flip_output):
        # flip output horizontally
        flip_output = proc.image_fliplr(flip_output.numpy())
        return torch.from_numpy(flip_output).float()

    # region function：新加 20230326。tensor水平翻转
    # endregion
    @classmethod
    def fliplr_back_tensor(cls, flip_output):
        # 无论输入数据多少个维度，torch.fliplr()只对第二高的维度进行交换，即x=1的维度。
        if flip_output.ndim == 3:
            return torch.permute(torch.fliplr(torch.permute(flip_output, dims=(0, 2, 1))), dims=(0, 2, 1))
        elif flip_output.ndim == 4:
            return torch.permute(torch.fliplr(torch.permute(flip_output, dims=(0, 3, 2, 1))), dims=(0, 3, 2, 1))

    # region function：随机加噪 -- 按比例随机去均值
    # params：
    #   （1）imgMap：imgMap对象（C*H*W）
    #   （2）prob：触发概率
    # return：
    #   imgMap：操作后的imgMap对象（C*H*W）
    # endregion
    @classmethod
    def noisy_mean(cls, imgMap, prob=0.5):
        if random.random() <= prob:
            mu = imgMap.mean()
            imgMap = random.uniform(0.8, 1.2) * (imgMap - mu) + mu
            imgMap.add_(random.uniform(-0.2, 0.2)).clamp_(0, 1)
        return imgMap
