# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from utils.process import ProcessUtils as proc
from utils.augment import AugmentUtils as aug
from utils.udaap.utils_augment import Augment


class MultiDataset(torch.utils.data.Dataset):
    def update(self, pseudoArrays):
        self.dataArray = copy.deepcopy(self.dataArray_reset)
        for paIdx, pseudoArray in enumerate(pseudoArrays):
            for pseudoItem in pseudoArray:
                if pseudoItem["enable"] > 0:
                    kIdx = pseudoItem["kpID"].split("_")[-1]
                    kIdx_len = len(kIdx)
                    imageID = pseudoItem["kpID"][0:(-1-kIdx_len)]
                    # imageID, kIdx = pseudoItem["kpID"].split("_")[-1]
                    dataItem = [item for item in self.dataArray if item["imageID"] == imageID][0]
                    dataItem["kps"][paIdx][int(kIdx)] = [pseudoItem["coord"][0], pseudoItem["coord"][1], pseudoItem["enable"]]
                    dataItem["islabeled"][paIdx] = 1

    def __init__(self, dsType, dataArray, means, stds, augCount=1, gtCount=1, isAug=False, isDraw=False, **kwargs):
        self.augCount, self.gtCount = augCount, gtCount
        self.dataArray = self._dataArray_init(dataArray, self.gtCount)
        self.dataArray_reset = copy.deepcopy(self.dataArray)
        self.dsType, self.means, self.stds, self.isAug, self.isDraw = dsType, means, stds, isAug, isDraw
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
        img, kpsArray, kps_test, imageID = proc.image_load(annObj["imagePath"]), annObj["kps"], annObj["kps_test"], annObj["imageID"]  # H*W*C
        img, kpsArray, _ = proc.image_resize_mulKps(img, kpsArray, self.inpRes)  # H*W*C
        _, kps_test, _ = proc.image_resize(img, kps_test, self.inpRes)  # 测试用。用于验证选用的伪标签的质量。
        test_kpsMap = torch.from_numpy(np.array(kps_test).astype(np.float32))  # 测试用。用于验证选用的伪标签的质量。
        # endregion

        # region 2. 生成原样本对应数据
        islabeled = [torch.tensor(annObj["islabeled"][aIdx] == 1) for aIdx in range(self.gtCount)]
        ori_imgMap = proc.image_np2tensor(img)  # H*W*C ==> C*H*W
        ori_kpsMaps = [torch.from_numpy(np.array(kps).astype(np.float32)) for kps in kpsArray]
        ori_heatmaps, ori_kpsWeights = [], []
        for mdsIdx in range(self.gtCount):
            ori_heatmap, ori_kpsMap = proc.kps_heatmap(ori_kpsMaps[mdsIdx], ori_imgMap.shape, self.inpRes, self.outRes)
            ori_heatmaps.append(ori_heatmap)
            ori_kpsMaps[mdsIdx] = ori_kpsMap
            ori_kpsWeights.append(ori_kpsMap[:, 2].clone())
        ori_angle = torch.tensor(0.)
        ori_scale = torch.tensor(self.inpRes / 200.0)
        ori_cenMap = torch.tensor(proc.center_calculate(img))
        ori_isflip = torch.tensor(False)
        # endregion

        # region 3. 生成增广样本对应数据
        # region 3.1 增广样本初始化
        augs_imgMap = [ori_imgMap.clone() for idx in range(self.augCount)]
        augs_kpsMaps = [copy.deepcopy(ori_kpsMaps) for idx in range(self.augCount)]
        augs_angle = [ori_angle.clone() for idx in range(self.augCount)]
        augs_scale = [ori_scale.clone() for idx in range(self.augCount)]
        augs_cenMap = [ori_cenMap.clone() for idx in range(self.augCount)]
        augs_isflip = [torch.tensor(False) for idx in range(self.augCount)]
        augs_heatmaps, augs_kpsWeights, augs_warpmat = [], [], []
        # endregion

        # region 3.2 数据增强
        if self.isDraw:
            draw_imgMap_flips, draw_kpsMaps_flips, draw_imgMap_affines, draw_kpsMaps_affines, draw_imgMap_occlusions = [], [], [], [], []
        for idx in range(self.augCount):
            # 数据准备
            aug_imgMap, aug_kpsMaps, aug_angle = augs_imgMap[idx], augs_kpsMaps[idx], augs_angle[idx]
            aug_scale, aug_cenMap, aug_isflip = augs_scale[idx], augs_cenMap[idx], augs_isflip[idx]

            # 数据增强
            if self.isAug:
                # 随机水平翻转
                aug_imgMap, aug_kpsMaps, aug_cenMap, aug_isflip = aug.fliplr_mulKps(aug_imgMap, aug_kpsMaps, aug_cenMap, prob=0.5)
                # 数据增强测试用 -- 存储随机水平翻转版本数据
                if self.isDraw:
                    draw_imgMap_flips.append(aug_imgMap.clone())
                    draw_kpsMaps_flips.append(copy.deepcopy(aug_kpsMaps))

                # 随机加噪（随机比例去均值）
                aug_imgMap = aug.noisy_mean(aug_imgMap)

                # 随机仿射变换（随机缩放、随机旋转）
                aug_imgMap, aug_kpsMaps, aug_scale, aug_angle = aug.affine_mulKps(aug_imgMap, aug_kpsMaps, aug_cenMap, aug_scale, self.sf, aug_angle, self.rf, [self.inpRes, self.inpRes])
                # 数据增强测试用 -- 存储随机仿射变换版本数据
                if self.isDraw:
                    draw_imgMap_affines.append(aug_imgMap.clone())
                    draw_kpsMaps_affines.append(copy.deepcopy(aug_kpsMaps))

                # 随机遮挡
                if self.useOcc:
                    aug_imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(aug_imgMap))
                    aug_imgMap = proc.image_np2tensor(aug_imgMap)
                    # 数据增强测试用 -- 存储随机遮挡版本数据
                    if self.isDraw: draw_imgMap_occlusions.append(aug_imgMap.clone())

            # 数据处理
            # 图像RGB通道去均值（C*H*W）
            aug_imgMap = proc.image_colorNorm(aug_imgMap, self.means, self.stds)
            # 生成kpsMap对应的heatmap
            aug_heatmaps, aug_kpsMaps = proc.kps_heatmap_mulKps(aug_kpsMaps, aug_imgMap.shape, self.inpRes, self.outRes)
            aug_kpsWeights = [aug_kpsMap[:, 2].clone() for aug_kpsMap in aug_kpsMaps]
            aug_warpmat = aug.affine_getWarpmat(-aug_angle, 1/aug_scale, matrixRes=[self.inpRes, self.inpRes])

            # 数据归档
            augs_imgMap[idx], augs_kpsMaps[idx], augs_angle[idx] = aug_imgMap, aug_kpsMaps, aug_angle
            augs_scale[idx], augs_cenMap[idx], augs_isflip[idx] = aug_scale, aug_cenMap, aug_isflip
            augs_heatmaps.append(aug_heatmaps)
            augs_kpsWeights.append(aug_kpsWeights)
            augs_warpmat.append(aug_warpmat)
        # endregion
        # endregion

        # region 4.数据增强测试
        if self.isDraw:
            for aIdx in range(self.augCount):
                for mdsIdx in range(self.gtCount):
                    # region 4.1 输出原图（含关键点）
                    draw_kpsMap = ori_kpsMaps[mdsIdx] if islabeled[mdsIdx] else test_kpsMap
                    self._draw_testImage(imageID, "aug{}_gt{}_01_ori".format(aIdx+1, mdsIdx+1), ori_imgMap.clone(), draw_kpsMap)

                    if islabeled[mdsIdx]:
                        draw2_kpsMap = torch.ones((ori_kpsWeights[mdsIdx].size(0), 3))
                        draw2_kpsMap[:, 0:2] = proc.kps_fromHeatmap(ori_heatmaps[mdsIdx].detach().cpu(), ori_cenMap, ori_scale, [self.outRes, self.outRes], mode="single")  # 使用ori_scale
                        draw2_kpsMap[:, 2] = ori_kpsWeights[mdsIdx]
                        self._draw_testImage(imageID, "aug{}_gt{}_02_ori_heatmap".format(aIdx+1, mdsIdx+1), ori_imgMap.clone(), draw2_kpsMap)
                    # endregion

                    # region 4.2 输出水平翻转图
                    if islabeled[mdsIdx] and self.isAug and self.useFlip:
                        self._draw_testImage(imageID, "aug{}_gt{}_03_flip".format(aIdx+1, mdsIdx+1), draw_imgMap_flips[aIdx], draw_kpsMaps_flips[aIdx][mdsIdx])
                    # endregion

                    # region 4.3 输出变换后图（含关键点）
                    if islabeled[mdsIdx] and self.isAug:
                        self._draw_testImage(imageID, "aug{}_gt{}_04_affine".format(aIdx+1, mdsIdx+1), draw_imgMap_affines[aIdx], draw_kpsMaps_affines[aIdx][mdsIdx])
                    # endregion

                    # region 4.4 输出遮挡后图（含关键点）
                    if islabeled[mdsIdx] and self.isAug and self.useOcc:
                        self._draw_testImage(imageID, "aug{}_gt{}_05_occlus".format(aIdx+1, mdsIdx+1), draw_imgMap_occlusions[aIdx], draw_kpsMaps_affines[aIdx][mdsIdx])
                    # endregion

                    # region 4.5 输出warpmat变换后图（含关键点）
                    if islabeled[mdsIdx] and self.isAug:
                        # region 反向仿射变换
                        draw_heatmap = augs_heatmaps[aIdx][mdsIdx].clone().unsqueeze(0)
                        affine_grid = F.affine_grid(augs_warpmat[aIdx].unsqueeze(0), draw_heatmap.size(), align_corners=True)
                        draw_heatmap = F.grid_sample(draw_heatmap, affine_grid, align_corners=True).squeeze(0)
                        # 进行反向水平翻转
                        if augs_isflip[aIdx]: draw_heatmap = aug.fliplr_back_tensor(draw_heatmap)
                        # 从heatmap中获得关键点
                        draw_kpsMap = torch.ones((augs_kpsWeights[aIdx][mdsIdx].size(0), 3))
                        draw_kpsMap[:, 0:2] = proc.kps_fromHeatmap(draw_heatmap, augs_cenMap[aIdx], torch.tensor(1), [self.outRes, self.outRes], mode="single")  # 使用aug_scale ==> 改为 torch.tensor(1)
                        draw_kpsMap[:, 2] = augs_kpsWeights[aIdx][mdsIdx]
                        # endregion
                        self._draw_testImage(imageID, "aug{}_gt{}_06_warpmat".format(aIdx+1, mdsIdx+1), ori_imgMap.clone(), draw_kpsMap)
                    # endregion

                    # region 4.6 输出最终图（含关键点）-- 只输出标记样本的数据增广样本
                    if islabeled[mdsIdx] and self.isAug:
                        draw_kpsMap = torch.ones((augs_kpsWeights[aIdx][mdsIdx].size(0), 3))
                        draw_kpsMap[:, 0:2] = proc.kps_fromHeatmap(augs_heatmaps[aIdx][mdsIdx].detach().cpu(), augs_cenMap[aIdx], ori_scale, [self.outRes, self.outRes], mode="single")  # 使用ori_scale
                        draw_kpsMap[:, 2] = augs_kpsWeights[aIdx][mdsIdx]
                        self._draw_testImage(imageID, "aug{}_gt{}_07_heatmap".format(aIdx+1, mdsIdx+1), augs_imgMap[aIdx].clone(), draw_kpsMap)
                        self._draw_testImage(imageID, "aug{}_gt{}_08_final".format(aIdx+1, mdsIdx+1), augs_imgMap[aIdx].clone(), augs_kpsMaps[aIdx][mdsIdx])
                    # endregion

                    # region 4.7 输出增广样本的image的反向变换图
                    if self.isAug:
                        # region 反向仿射变换
                        draw_img = augs_imgMap[aIdx].clone().unsqueeze(0)
                        affine_grid = F.affine_grid(augs_warpmat[aIdx].unsqueeze(0), draw_img.size(), align_corners=True)
                        draw_img = F.grid_sample(draw_img, affine_grid, align_corners=True).squeeze(0)
                        # 进行反向水平翻转
                        if augs_isflip[aIdx]: draw_img = aug.fliplr_back_tensor(draw_img)
                        self._draw_testImage(imageID, "aug{}_gt{}_09_imgWarpmat".format(aIdx+1, mdsIdx+1), draw_img, [])
                    # endregion
        # endregion

        # augs_heatmaps, augs_kpsMaps, augs_kpsWeights：[augNum, mdsNum, ...]
        # augs_warpmat, augs_cenMap, augs_angleaugs_scale, augs_isflip：[augNum, ...]
        meta = {"imageID": annObj["imageID"], "imagePath": annObj["imagePath"], "islabeled": islabeled,
                "warpmat": augs_warpmat, "kpsMaps": augs_kpsMaps, 'kpsWeights': augs_kpsWeights, "center": augs_cenMap, "angle": augs_angle, "scale": augs_scale, "isflip": augs_isflip,
                "ori_imgMap": ori_imgMap, "ori_kpsMap": ori_kpsMaps, 'ori_kpsWeight': ori_kpsWeights, "ori_cenMap": ori_cenMap, "ori_angle": ori_angle, "ori_scale": ori_scale, "ori_isflip": ori_isflip,
                "kpsMap_test": test_kpsMap}
        return augs_imgMap, augs_heatmaps, meta

    def __len__(self):
        return len(self.dataArray)

    def _dataArray_init(self, dataArray, gtCount):
        dataArray_res = copy.deepcopy(dataArray)
        for dataItem in dataArray_res:
            islabeled, kps = dataItem["islabeled"], dataItem["kps"]
            dataItem["islabeled"] = [islabeled for i in range(gtCount)]
            dataItem["kps"] = [copy.deepcopy(kps) for i in range(gtCount)]
        return dataArray_res

    def _draw_testImage(self, imageID, stepID, imgMap, kpsMap):
        img_draw = proc.image_tensor2np(imgMap.detach().cpu().data * 255).astype(np.uint8)
        if len(kpsMap) > 0:
            kps_draw = kpsMap.detach().cpu().data.numpy().astype(int).tolist()
            for kIdx, kp in enumerate(kps_draw):
                if kp[2] > 0:
                    img_draw = proc.draw_point(img_draw, kp[0:2], radius=3, thickness=-1, color=(0, 95, 191))
        proc.image_save(img_draw, "{}/draw/dataset/{}/{}_{}.{}".format(self.basePath, self.dsType, imageID, stepID, self.imgType))
        del img_draw
