# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
import math
from itertools import combinations as comb

from utils.base.comm import CommUtils as comm
from .udaap.transforms import transform_preds
from .udaap.evaluation import final_preds


class ProcessUtils:
    def __init__(self):
        pass

    @classmethod
    def features_cov(cls, inp1, inp2):
        bs, n, c, h, w = inp1.size()
        f1 = inp1.clone().view(bs, n, c, h * w)
        f2 = inp2.clone().view(bs, n, c, h * w)
        vecs = torch.stack([f1, f2], -1)
        cov_matrix = cls.torch_cov(cls, vecs)
        return torch.mean(torch.mean(torch.mean(torch.abs(cov_matrix[:, :, :, 0, 1]), dim=-1), dim=-1), dim=-1), bs*n*c

    def torch_cov(self, input_vec):
        x = input_vec - torch.mean(input_vec, dim=-2).unsqueeze(-2)
        x_T = torch.transpose(x.clone(), -2, -1)  # [bs, n, c, 2, h*w]
        cov_matrix = torch.matmul(x_T, x) / (x.shape[-2] - 1)
        return cov_matrix

    @classmethod
    def feature_mixture_across_epoch(cls, featuresArray, args):
        tNum = len(featuresArray) if args.maxEpoCount >= len(featuresArray) else args.maxEpoCount
        wArray = [cls._feature_mixture_weight(cls, tIdx) for tIdx in range(tNum)]
        val_molecule, val_denominator = 0., sum(wArray)
        for tIdx in range(tNum):
            val_molecule += wArray[tIdx] * featuresArray[len(featuresArray) - 1 - tIdx].detach()  # detach有问题，待调研
        return val_molecule/val_denominator

    def _feature_mixture_weight(self, epo, l=10):
        t = epo - l
        return (1 - comm.math_signal(t) + comm.math_signal(t)*math.exp(-comm.math_signal(t)*t))/2

    # region function：计算两点像素距离
    # params:
    #   （1）coord1：坐标1
    #   （2）coord2：坐标2
    # return: 两点像素距离
    # endregion
    @classmethod
    def coord_distance(cls, coord1, coord2):
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

    @classmethod
    def coord_avgDistance(cls, coords):
        coords_comb = comb(coords, 2)
        dist_sum, dist_n = 0., 0
        while True:
            try:
                coords_pair = next(coords_comb)
                dist = cls.coord_distance(coords_pair[0], coords_pair[1])
                dist_sum += dist
                dist_n += 1
            except StopIteration:
                break
        return dist_sum/dist_n

    # region function：将Coco风格的bbox格式化为[leftTop, rightBottom]，并取整
    # params: box对象 [minX, minY, width, height]
    # return: 标准化后的box对象
    # endregion
    @classmethod
    def box_cocoStandard(cls, bbox):
        minX, minY, width, height = bbox
        lt = [minX, minY]
        rb = [minX + width, minY + height]
        return [lt, rb]

    # region function：读取图像
    # params：
    #   （1）pathname：图像路径
    # return：image对象（H*W*C）
    # endregion
    @classmethod
    def image_load(cls, pathname):
        return np.array(cv2.imread(pathname), dtype=np.float32)

    # region function：保存图像
    # params：
    #   （1）img：image对象（H*W*C）
    #   （2）pathname：图像路径
    #   （3）compression：图片压缩值，0为高清，9为最大压缩（压缩时间长），默认3
    # endregion
    @classmethod
    def image_save(cls, img, pathname, compression=0):
        # 创建路径
        folderPath = os.path.split(pathname)[0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        cv2.imwrite(pathname, img, [cv2.IMWRITE_PNG_COMPRESSION, compression])  # 压缩值，0为高清，9为最大压缩（压缩时间长），默认3.

    # region function：调整图像大小（保持纵横比）
    # params：
    #   （1）img：image对象（H*W*C）
    #   （2）kps：keypoint集合
    #   （3）inpRes：模型输入尺寸
    # return：
    #   （1）img：操作后的image对象（H*W*C）
    #   （2）kps：操作后的keypoint集合
    #   （3）scale：缩放比例（target/original）
    # endregion
    @classmethod
    def image_resize(cls, img, kps, inpRes):
        h, w, _ = img.shape
        scale = [inpRes / w, inpRes / h]
        img = cv2.resize(img, (inpRes, inpRes))
        kps = [[kp[0] * scale[0], kp[1] * scale[1], kp[2]] for kp in kps]
        return img, kps, scale

    # region function：调整图像大小（保持纵横比）
    # params：
    #   （1）img：image对象（H*W*C）
    #   （2）kps：keypoint集合
    #   （3）inpRes：模型输入尺寸
    # return：
    #   （1）img：操作后的image对象（H*W*C）
    #   （2）kps：操作后的keypoint集合
    #   （3）scale：缩放比例（target/original）
    # endregion
    @classmethod
    def image_resize_mulKps(cls, img, kpsArray, inpRes):
        h, w, _ = img.shape
        scale = [inpRes / w, inpRes / h]
        img = cv2.resize(img, (inpRes, inpRes))
        kpsArray_new = []
        for kps in kpsArray:
            kpsArray_new.append([[kp[0] * scale[0], kp[1] * scale[1], kp[2]] for kp in kps])
        return img, kpsArray_new, scale

    # region function：图像颜色正则化
    # params：
    #   （1）img：image对象（C*H*W）
    #   （2）means：均值
    #   （3）stds：方差
    #   （4）useStd：是否使用方差
    # return：
    #   （1）img：操作后的image对象（C*H*W）
    # endregion
    @classmethod
    def image_colorNorm(cls, img, means, stds, useStd=False):
        if img.size(0) == 1:  # 黑白图处理
            img = img.repeat(3, 1, 1)

        for t, m, s in zip(img, means, stds):  # 彩色图处理
            t.sub_(m)  # 去均值，未对方差进行处理。
            if useStd:
                t.div_(s)
        return img

    # region function：图像转换 -- numpy (H*W*C) to tensor (C*H*W)
    # params：
    #   （1）imgNdarry：ndarry类型的image对象（H*W*C）
    #   （2）inpRes：输入特征分辨率
    # return：
    #   （1）imgMap：tensor类型的image对象（C*H*W）
    # endregion
    @classmethod
    def image_np2tensor(cls, imgNdarry):
        if imgNdarry.shape[0] != 1 and imgNdarry.shape[0] != 3:
            imgNdarry = np.transpose(imgNdarry, (2, 0, 1))  # H*W*C ==> C*H*W
        imgMap = torch.from_numpy(imgNdarry.astype(np.float32))
        if imgMap.max() > 1:
            imgMap /= 255
        return imgMap

    # region function：图像转换 -- tensor (C*H*W) to numpy (H*W*C)
    # params：
    #   （1）imgMap：tensor类型的image对象（C*H*W）
    # return：
    #   （1）imgNdarry：ndarry类型的image对象（H*W*C）
    # endregion
    @classmethod
    def image_tensor2np(cls, imgMap):
        if not torch.is_tensor(imgMap): return None
        imgNdarry = imgMap.detach().cpu().numpy()
        if imgNdarry.shape[0] == 1 or imgNdarry.shape[0] == 3:
            imgNdarry = np.transpose(imgNdarry, (1, 2, 0))  # C*H*W ==> H*W*C
            imgNdarry = np.ascontiguousarray(imgNdarry)
        return imgNdarry

    # region function：图像水平翻转 -- tensor (C*H*W) to numpy (H*W*C)
    # params：
    #   （1）imgNdarry：ndarry类型的image对象（H*W*C）
    # return：
    #   （1）imgNdarry：ndarry类型的image对象（H*W*C）
    # endregion
    @classmethod
    def image_fliplr(cls, imgNdarry):
        if imgNdarry.ndim == 3:
            # np.fliplr 左右翻转
            imgNdarry = np.transpose(np.fliplr(np.transpose(imgNdarry, (0, 2, 1))), (0, 2, 1))
        elif imgNdarry.ndim == 4:
            for i in range(imgNdarry.shape[0]):
                imgNdarry[i] = np.transpose(np.fliplr(np.transpose(imgNdarry[i], (0, 2, 1))), (0, 2, 1))
        return imgNdarry.astype(float)

    # region function：计算中心点
    # params：
    #   （1）img：image对象（H*W*C）
    #   （2）kps：keypoint集合
    #   （3）cType：中心计算方式（"imgCenter", "kpsCenter"）
    # return：
    #   （1）center：中心点坐标
    # endregion
    @classmethod
    def center_calculate(cls, img, kps=None, cType="imgCenter"):
        h, w, _ = img.shape
        if cType == "imgCenter":
            return [int(w / 2), int(h / 2)]
        elif cType == "kpsCenter" and kps is not None:
            c, n = [0, 0], 0
            for kp in kps:
                if kp[2] == 0: continue
                c[0] += kp[0]
                c[1] += kp[1]
                n += 1
            return [int(c[0] / n), int(c[1] / n)]

    # region function：关键点水平翻转
    # params：
    #   （1）kps：keypoint集合
    #   （2）imgWidth：图像宽度
    # return：
    #   （1）kps：操作后的keypoint集合
    # endregion
    @classmethod
    def kps_fliplr(cls, kpsMap, imgWidth):
        # 对坐标值的修改（Flip horizontal）
        kpsMap[:, 0] = imgWidth - kpsMap[:, 0]
        return kpsMap

    # region function：生成keypoints对应heatmap
    # params:
    #   （1）kps：keypoint集合
    #   （2）imgShape：图像大小（C*H*W）
    #   （3）inpRes：输入特征分辨率
    #   （4）outRes：输出特征分辨率
    # return: heatmap对象
    # endregion
    @classmethod
    def kps_heatmap(cls, kpsMap, imgShape, inpRes, outRes, kernelSize=3.0, sigma=1.0):
        _, h, w = imgShape  # C*H*W
        stride = inpRes / outRes
        sizeH, sizeW = int(h / stride), int(w / stride)  # 计算HeatMap尺寸
        kpsCount = len(kpsMap)
        sigma *= kernelSize
        # 将HeatMap大小设置网络最小分辨率
        heatmap = np.zeros((sizeH, sizeW, kpsCount), dtype=np.float32)
        for kIdx in range(kpsCount):
            # 检查高斯函数的任意部分是否在范围内
            kp_int = kpsMap[kIdx].to(torch.int32)
            ul = [int(kp_int[0] - sigma), int(kp_int[1] - sigma)]
            br = [int(kp_int[0] + sigma + 1), int(kp_int[1] + sigma + 1)]
            vis = 0 if (br[0] >= w or br[1] >= h or ul[0] < 0 or ul[1] < 0) else 1
            kpsMap[kIdx][2] *= vis

            # 将keypoints转化至指定分辨率下
            x = int(kpsMap[kIdx][0]) * 1.0 / stride
            y = int(kpsMap[kIdx][1]) * 1.0 / stride
            kernel = cls.heatmap_gaussian(sizeH, sizeW, center=[x, y], sigma=sigma)
            # 边缘修正
            kernel[kernel > 1] = 1
            kernel[kernel < 0.01] = 0
            heatmap[:, :, kIdx] = kernel
        heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1)))
        return heatmap.float(), kpsMap

    # region function：生成keypoints对应heatmap
    # params:
    #   （1）kps：keypoint集合
    #   （2）imgShape：图像大小（C*H*W）
    #   （3）inpRes：输入特征分辨率
    #   （4）outRes：输出特征分辨率
    # return: heatmap对象
    # endregion
    @classmethod
    def kps_heatmap_mulKps(cls, kpsMapArray, imgShape, inpRes, outRes, kernelSize=3.0, sigma=1.0):
        _, h, w = imgShape  # C*H*W
        stride = inpRes / outRes
        sizeH, sizeW = int(h / stride), int(w / stride)  # 计算HeatMap尺寸
        sigma *= kernelSize
        heatmapArray, kpsMapArray_new = [], []
        for kpsMap in kpsMapArray:
            kpsCount = len(kpsMap)
            # 将HeatMap大小设置网络最小分辨率
            heatmap = np.zeros((sizeH, sizeW, kpsCount), dtype=np.float32)
            for kIdx in range(kpsCount):
                # 检查高斯函数的任意部分是否在范围内
                kp_int = kpsMap[kIdx].to(torch.int32)
                ul = [int(kp_int[0] - sigma), int(kp_int[1] - sigma)]
                br = [int(kp_int[0] + sigma + 1), int(kp_int[1] + sigma + 1)]
                vis = 0 if (br[0] >= w or br[1] >= h or ul[0] < 0 or ul[1] < 0) else 1
                kpsMap[kIdx][2] *= vis

                # 将keypoints转化至指定分辨率下
                x = int(kpsMap[kIdx][0]) * 1.0 / stride
                y = int(kpsMap[kIdx][1]) * 1.0 / stride
                kernel = cls.heatmap_gaussian(sizeH, sizeW, center=[x, y], sigma=sigma)
                # 边缘修正
                kernel[kernel > 1] = 1
                kernel[kernel < 0.01] = 0
                heatmap[:, :, kIdx] = kernel
            heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1)))
            heatmapArray.append(heatmap.float())
            kpsMapArray_new.append(kpsMap)
        return heatmapArray, kpsMapArray_new

    @classmethod
    def kps_fromHeatmap(cls, heatmap, cenMap, scale, res, mode="batch"):
        if mode == "single":
            return final_preds(heatmap.unsqueeze(0), cenMap.unsqueeze(0), scale.unsqueeze(0), res)[0]
        elif mode == "batch":
            preds = final_preds(heatmap, cenMap, scale, res)
            scores = torch.from_numpy(np.max(heatmap.detach().cpu().numpy(), axis=(2, 3)).astype(np.float32))  # detach待调研
            return preds, scores

    @classmethod
    def kps_fromHeatmap_mul(cls, multiOuts, cenMap, scale, res):
        mc, bs, k, _, _ = multiOuts.shape
        predsMulti = torch.stack([final_preds(multiOuts[mcIdx], cenMap, scale, res) for mcIdx in range(mc)], 0)
        predsMean = torch.mean(predsMulti, dim=0)
        scoresMulti = torch.from_numpy(np.max(multiOuts.detach().cpu().numpy(), axis=(3, 4)).astype(np.float32))  # detach待调研
        scoresMean = torch.mean(scoresMulti, dim=0)
        return predsMulti, predsMean, scoresMulti, scoresMean

    # region function：从heatmap中获得keypoint
    # params:
    #   （1）heatmap：输出heatmap
    #   （2）imgShape：图像大小（C*H*W）
    # return: kps集合
    # endregion
    @classmethod
    def kps_fromHeatmap2(cls, heatmap, cenMap, scale, res):
        # region 1.获取关键点坐标（64*64）
        k, w, h = heatmap.size()
        heatmap_row = heatmap.view(k, -1)  # 3维向量（k*w*h）==>2维向量（k*(w*h)）
        confidences_max, coords_max = torch.max(heatmap_row, 1)  # 在index=1的维度取最大值及其索引。
        confidences_max = confidences_max.view(k, 1)  # 在index=1处添加一个维度，形成：torch.Size([1, 18, 1])
        coords_max = coords_max.view(k, 1) + 1  # 在index=1处添加一个维度，形成：torch.Size([1, 18, 1])。此外，index从0开始，所以所有索引加1。
        # tensor.repeat()：https://blog.csdn.net/qq_29695701/article/details/89763168
        preds = coords_max.repeat(1, 2).float()  # 在index=1的维度上增加一个相同的值。由idx's torch.Size([1, 18, 1]) ==> preds's torch.Size([1, 18, 1])
        # 将各点预测最高点的索引集合从1*(w*h)的特征图映射到w*h的特征图中
        preds[:, 0] = (preds[:, 0] - 1) % h + 1  # 在w*h的特征图中，计算其列数（y）
        preds[:, 1] = torch.floor((preds[:, 1] - 1) / h) + 1  # 在w*h的特征图中，计算其行数（x）
        # tensor.gt(0)：各项与0比，大于0则为True，小于则为False
        pred_mask = confidences_max.gt(0).repeat(1, 2).float()  # 与preds对应，形成一个由confidence score量化的一个是否使用的mask（小于0的则认为不可信，不采用）
        preds *= pred_mask
        # endregion

        # region 2.修正关键点坐标（64*64）
        for p in range(preds.size(1)):
            hm = heatmap[p]  # 关键点对应的64*64特征图。
            px = int(math.floor(preds[p][0]))  # 关键点的位置中的x。
            py = int(math.floor(preds[p][1]))  # 关键点的位置中的y。
            if 1 < px < res[0] and 1 < py < res[1]:  # 验证预测位置的合法性
                # 计算预测位置的四周值的差值。
                diff = torch.from_numpy(
                    np.array([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]]))
                preds[p] += diff.sign() * .25  # 对预测位置的四周值做了一个附加的评估，对应调整下位置。 diff.sign()：大于0则为1，小于0则为-1
        preds += 0.5  # 没看懂，为什么加。
        # endregion

        # region 3.映射到原图坐标（256*256） -- 待调研，先用着。
        preds = transform_preds(preds, cenMap, scale, res)
        # endregion

        return preds

    @classmethod
    def kps_getLabeledCount(cls, kpsGate):
        return len([item for item in kpsGate.detach().reshape(-1).cpu().data.numpy() if item > 0])

    # region function：生成高斯kernel，用于生成kps对应的heatmap
    # params:
    #   （1）h：图像height
    #   （2）w：图像width
    #   （3）center：中心点
    #   （4）sigma：sigma系数
    # return: kernel对象
    # endregion
    @classmethod
    def heatmap_gaussian(cls, h, w, center, sigma=3.0):
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        D2 = (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)


    # region function：打点
    # params:
    #   （1）img：image对象（H*W*C）
    #   （2）coord：点坐标
    #   （3）radius：半径
    #   （4）thickness：边线粗
    #   （5）color：颜色
    # return:
    #   （1）img：绘制后的image对象（H*W*C）
    # endregion
    @classmethod
    def draw_point(cls, img, coord, color=(0, 95, 191), radius=3, thickness=-1, text=None, textScale=1.0, textColor=(255, 255, 255)):
        img, x, y = img.astype(int), round(coord[0]), round(coord[1])
        if x > 1 and y > 1:
            cv2.circle(img, (x, y), color=color, radius=radius, thickness=thickness)
            if text is not None:
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, textScale, textColor, 2)
        return img
