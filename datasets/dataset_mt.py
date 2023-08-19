# -*- coding: utf-8 -*-
import os
import random
import copy
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from utils.process import ProcessUtils as proc
from utils.augment import AugmentUtils as aug
from utils.udaap.utils_augment import Augment


# 用于Mean-Teacher结构的数据供给。
# 特性是：数据增强后，未提供teacher的样本标签数据（既，未提供kpsHeatmap_ema）
class MTDataset(torch.utils.data.Dataset):
    def update(self, pseudoArray):
        self.dataArray = copy.deepcopy(self.dataArray_reset)
        for pseudoItem in pseudoArray:
            if pseudoItem["enable"] > 0:
                kIdx = pseudoItem["kpID"].split("_")[-1]
                kIdx_len = len(kIdx)
                imageID = pseudoItem["kpID"][0:(-1-kIdx_len)]
                # imageID, kIdx = pseudoItem["kpID"].split("_")
                dataItem = [item for item in self.dataArray if item["imageID"] == imageID][0]
                dataItem["kps"][int(kIdx)] = [pseudoItem["coord"][0], pseudoItem["coord"][1], pseudoItem["enable"]]

    def __init__(self, dsType, dataArray, means, stds, isAug=False, isDraw=False, **kwargs):
        self.dsType, self.dataArray, self.means, self.stds, self.isAug, self.isDraw = dsType, copy.deepcopy(dataArray), means, stds, isAug, isDraw
        self.dataArray_reset = copy.deepcopy(dataArray)
        self.inpRes, self.outRes, self.basePath, self.imgType = kwargs['inpRes'], kwargs['outRes'], kwargs['basePath'], kwargs['imgType']
        self.useFlip, self.useOcc, self.sf, self.rf = kwargs['useFlip'], kwargs['useOcclusion'], kwargs['scaleRange'], kwargs['rotRange']
        self.useOcc_ema, self.sf_ema, self.rf_ema = kwargs['useOcclusion_ema'], kwargs['scaleRange_ema'], kwargs['rotRange_ema']
        if self.isAug:
            if self.useOcc:
                self.augmentor = Augment(num_occluder=kwargs['numOccluder'])  # student专用。从voc2012数据集中拿到一些mask形状，来遮挡图像。
            if self.useOcc_ema:
                self.augmentor_ema = Augment(num_occluder=kwargs['numOccluder_ema'])  # teacher专用。从voc2012数据集中拿到一些mask形状，来遮挡图像。

    def __getitem__(self, idx):
        # region 1 数据预处理（含图像resize）
        annObj = self.dataArray[idx]
        img, kps, kps_test, imageID = proc.image_load(annObj["imagePath"]), annObj["kps"], annObj["kps_test"], annObj["imageID"]  # H*W*C
        img, kps, _ = proc.image_resize(img, kps, self.inpRes)  # H*W*C
        _, kps_test, _ = proc.image_resize(img, kps_test, self.inpRes)  # 测试用。用于验证选用的伪标签的质量。
        test_kpsMap = torch.from_numpy(np.array(kps_test).astype(np.float32))  # 测试用。用于验证选用的伪标签的质量。
        # endregion

        # region 2. student数据组织
        islabeled = torch.tensor(annObj["islabeled"] == 1)
        ori_imgMap = proc.image_np2tensor(img)  # H*W*C ==> C*H*W
        ori_kpsMap = torch.from_numpy(np.array(kps).astype(np.float32))
        ori_heatmap, ori_kpsMap = proc.kps_heatmap(ori_kpsMap, ori_imgMap.shape, self.inpRes, self.outRes)
        ori_kpsWeight = ori_kpsMap[:, 2].clone()
        ori_angle = torch.tensor(0.)
        ori_scale = torch.tensor(self.inpRes / 200.0)
        ori_cenMap = torch.tensor(proc.center_calculate(img))
        # endregion

        # region 3. 增广样本初始化
        stu_imgMap = ori_imgMap.clone()
        stu_kpsMap = ori_kpsMap.clone()
        stu_angle = ori_angle.clone()
        stu_scale = ori_scale.clone()
        stu_cenMap = ori_cenMap.clone()
        stu_isflip = torch.tensor(False)
        # endregion

        # region 4. teacher数据组织
        ema_imgMap = stu_imgMap.clone()
        ema_kpsMap = stu_kpsMap.clone()
        ema_angle = stu_angle.clone()
        ema_scale = stu_scale.clone()
        ema_cenMap = stu_cenMap.clone()
        ema_isflip = torch.tensor(False)
        # endregion

        # region 5.数据增强
        if self.isAug:
            # 随机水平翻转
            stu_imgMap, stu_kpsMap, stu_cenMap, stu_isflip = aug.fliplr(stu_imgMap, stu_kpsMap, stu_cenMap, prob=0.5)
            ema_imgMap, ema_kpsMap, ema_cenMap, ema_isflip = aug.fliplr(ema_imgMap, ema_kpsMap, ema_cenMap, prob=0.5)
            if self.isDraw:
                draw_stu_imgMap_flip, draw_stu_kpsMap_flip = stu_imgMap.clone(), stu_kpsMap.clone()  # 测试用
                draw_ema_imgMap_flip, draw_ema_kpsMap_flip = ema_imgMap.clone(), ema_kpsMap.clone()  # 测试用

            # 随机加噪（随机比例去均值）
            stu_imgMap = aug.noisy_mean(stu_imgMap)
            ema_imgMap = aug.noisy_mean(ema_imgMap)

            # 随机仿射变换（随机缩放、随机旋转）
            stu_imgMap, stu_kpsMap, stu_scale, stu_angle = aug.affine(stu_imgMap, stu_kpsMap, stu_cenMap, stu_scale, self.sf, stu_angle, self.rf, [self.inpRes, self.inpRes])
            ema_imgMap, ema_kpsMap, ema_scale, ema_angle = aug.affine(ema_imgMap, ema_kpsMap, ema_cenMap, ema_scale, self.sf, ema_angle, self.rf, [self.inpRes, self.inpRes])
            if self.isDraw:
                draw_stu_imgMap_affine, draw_stu_kpsMap_affine = stu_imgMap.clone(), stu_kpsMap.clone()  # 测试用
                draw_ema_imgMap_affine, draw_ema_kpsMap_affine = ema_imgMap.clone(), ema_kpsMap.clone()  # 测试用

            # 随机遮挡
            if self.useOcc:
                stu_imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(stu_imgMap))
                stu_imgMap = proc.image_np2tensor(stu_imgMap)
                if self.isDraw:
                    draw_stu_imgMap_occlusion = stu_imgMap.clone()  # 测试用
            if self.useOcc_ema:
                ema_imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(ema_imgMap))
                ema_imgMap = proc.image_np2tensor(ema_imgMap)
                if self.isDraw:
                    draw_ema_imgMap_occlusion = ema_imgMap.clone()  # 测试用
        # endregion

        # region 6. 数据处理
        # 图像RGB通道去均值（C*H*W）
        stu_imgMap = proc.image_colorNorm(stu_imgMap, self.means, self.stds)
        ema_imgMap = proc.image_colorNorm(ema_imgMap, self.means, self.stds)
        # 生成kpsMap对应的heatmap
        stu_heatmap, stu_kpsMap = proc.kps_heatmap(stu_kpsMap, stu_imgMap.shape, self.inpRes, self.outRes)
        stu_kpsWeight = stu_kpsMap[:, 2].clone()
        stu_warpmat = aug.affine_getWarpmat(-stu_angle, 1/stu_scale, matrixRes=[self.inpRes, self.inpRes])
        # 生成kpsMap_ema对应的heatmap（测试用。仅用于测试数据变换的效果）
        ema_heatmap, ema_kpsMap = proc.kps_heatmap(ema_kpsMap, ema_imgMap.shape, self.inpRes, self.outRes)
        ema_kpsWeight = ema_kpsMap[:, 2].clone()
        ema_warpmat = aug.affine_getWarpmat(-ema_angle, 1/ema_scale, matrixRes=[self.inpRes, self.inpRes])
        # endregion

        # region 7.数据增强测试
        if self.isDraw:
            # region 7.1 输出原图（含关键点）
            draw_ori_kpsMap = ori_kpsMap if islabeled else test_kpsMap
            self._draw_testImage(imageID, "01_ori", ori_imgMap.clone(), draw_ori_kpsMap)

            if islabeled:
                draw2_kpsMap = torch.ones((ori_kpsWeight.size(0), 3))
                draw2_kpsMap[:, 0:2] = proc.kps_fromHeatmap(ori_heatmap.detach().cpu(), ori_cenMap, ori_scale, [self.outRes, self.outRes], mode="single")  # 使用ori_scale
                draw2_kpsMap[:, 2] = ori_kpsWeight
                self._draw_testImage(imageID, "02_ori_heatmap", ori_imgMap.clone(), draw2_kpsMap)
            # endregion

            # region 7.2 输出水平翻转图
            if islabeled and self.isAug and self.useFlip:
                self._draw_testImage(imageID, "stu_03_flip", draw_stu_imgMap_flip, draw_stu_kpsMap_flip)
                self._draw_testImage(imageID, "ema_03_flip", draw_ema_imgMap_flip, draw_ema_kpsMap_flip)
            # endregion

            # region 7.3 输出变换后图（含关键点）
            if islabeled and self.isAug:
                self._draw_testImage(imageID, "stu_04_affine", draw_stu_imgMap_affine, draw_stu_kpsMap_affine)
                self._draw_testImage(imageID, "ema_04_affine", draw_ema_imgMap_affine, draw_ema_kpsMap_affine)
            # endregion

            # region 7.4 输出遮挡后图（含关键点）
            if self.isAug and self.useOcc:
                self._draw_testImage(imageID, "stu_05_occlus", draw_stu_imgMap_occlusion, draw_stu_kpsMap_affine)
            if self.isAug and self.useOcc_ema:
                self._draw_testImage(imageID, "ema_05_occlus", draw_ema_imgMap_occlusion, draw_ema_kpsMap_affine)
            # endregion

            # region 7.5 输出warpmat变换后图（含关键点）
            if islabeled and self.isAug:
                draws_heatmap, draws_kpsMap, draws_kpsWeight = [stu_heatmap, ema_heatmap], [stu_kpsMap, ema_kpsMap], [stu_kpsWeight, ema_kpsWeight]
                draws_warpmat, draws_isflip, draws_cenMap, draws_scale = [stu_warpmat, ema_warpmat], [stu_isflip, ema_isflip],  [stu_cenMap, ema_cenMap],  [stu_scale, ema_scale]
                for idx, mark in enumerate(["stu", "ema"]):
                    # region 反向仿射变换
                    draw_heatmap = draws_heatmap[idx].clone().unsqueeze(0)
                    affine_grid = F.affine_grid(draws_warpmat[idx].unsqueeze(0), draw_heatmap.size(), align_corners=True)
                    draw_heatmap = F.grid_sample(draw_heatmap, affine_grid, align_corners=True).squeeze(0)
                    # 进行反向水平翻转
                    if draws_isflip[idx]: draw_heatmap = aug.fliplr_back_tensor(draw_heatmap)
                    # 从heatmap中获得关键点
                    draw_kpsMap = torch.ones((draws_kpsWeight[idx].size(0), 3))
                    draw_kpsMap[:, 0:2] = proc.kps_fromHeatmap(draw_heatmap, draws_cenMap[idx], torch.tensor(1), [self.outRes, self.outRes], mode="single")  # 使用aug_scale
                    draw_kpsMap[:, 2] = draws_kpsWeight[idx]
                    # endregion
                    self._draw_testImage(imageID, "{}_06_warpmat".format(mark), ori_imgMap.clone(), draw_kpsMap)
            # endregion

            # region 7.6 输出最终图（含关键点）-- 只输出标记样本的数据增广样本
            if islabeled and self.isAug:
                draw_kpsMap = torch.ones((stu_kpsWeight.size(0), 3))
                draw_kpsMap[:, 0:2] = proc.kps_fromHeatmap(stu_heatmap.detach().cpu(), stu_cenMap, ori_scale, [self.outRes, self.outRes], mode="single")  # 使用ori_scale
                draw_kpsMap[:, 2] = stu_kpsWeight
                self._draw_testImage(imageID, "stu_07_heatmap", stu_imgMap.clone(), draw_kpsMap)
                self._draw_testImage(imageID, "stu_08_final", stu_imgMap.clone(), stu_kpsMap)

                draw2_kpsMap = torch.ones((ema_kpsWeight.size(0), 3))
                draw2_kpsMap[:, 0:2] = proc.kps_fromHeatmap(ema_heatmap.detach().cpu(), ema_cenMap, ori_scale, [self.outRes, self.outRes], mode="single")  # 使用ori_scale
                draw2_kpsMap[:, 2] = ema_kpsWeight
                self._draw_testImage(imageID, "ema_07_heatmap", ema_imgMap.clone(), draw2_kpsMap)
                self._draw_testImage(imageID, "ema_08_final", ema_imgMap.clone(), ema_kpsMap)
            # endregion

            # region 7.7 输出增广样本的image的反向变换图
            if self.isAug:
                stu_draw_img = stu_imgMap.clone().unsqueeze(0)
                affine_grid = F.affine_grid(stu_warpmat.unsqueeze(0), stu_draw_img.size(), align_corners=True)
                stu_draw_img = F.grid_sample(stu_draw_img, affine_grid, align_corners=True).squeeze(0)
                # 进行反向水平翻转
                if stu_isflip: stu_draw_img = aug.fliplr_back_tensor(stu_draw_img)
                self._draw_testImage(imageID, "stu_09_imgWarpmat", stu_draw_img, [])

                ema_draw_img = ema_imgMap.clone().unsqueeze(0)
                affine_grid = F.affine_grid(ema_warpmat.unsqueeze(0), ema_draw_img.size(), align_corners=True)
                ema_draw_img = F.grid_sample(ema_draw_img, affine_grid, align_corners=True).squeeze(0)
                # 进行反向水平翻转
                if ema_isflip: ema_draw_img = aug.fliplr_back_tensor(ema_draw_img)
                self._draw_testImage(imageID, "ema_09_imgWarpmat", ema_draw_img, [])
            # endregion
        # endregion

        meta = {"imageID": annObj["imageID"], "imagePath": annObj["imagePath"], "islabeled": islabeled,
                "warpmat": stu_warpmat, "kpsMap": stu_kpsMap, 'kpsWeight': stu_kpsWeight, "center": stu_cenMap, "angle": stu_angle, "scale": stu_scale, "isflip": stu_isflip,
                "warpmat_ema": ema_warpmat, "kpsMap_ema": ema_kpsMap, 'kpsWeight_ema': ema_kpsWeight, "center_ema": ema_cenMap, "angle_ema": ema_angle, "scale_ema": ema_scale, "isflip_ema": ema_isflip}
        return stu_imgMap, stu_heatmap, ema_imgMap, meta

    def __len__(self):
        return len(self.dataArray)

    def _draw_testImage(self, imageID, stepID, imgMap, kpsMap):
        img_draw = proc.image_tensor2np(imgMap.detach().cpu().data * 255).astype(np.uint8)
        if len(kpsMap) > 0:
            kps_draw = kpsMap.detach().cpu().data.numpy().astype(int).tolist()
            for kIdx, kp in enumerate(kps_draw):
                if kp[2] > 0:
                    img_draw = proc.draw_point(img_draw, kp[0:2], radius=3, thickness=-1, color=(0, 95, 191))
        proc.image_save(img_draw, "{}/draw/dataset/{}/{}_{}.{}".format(self.basePath, self.dsType, imageID, stepID, self.imgType))
        del img_draw, kps_draw
