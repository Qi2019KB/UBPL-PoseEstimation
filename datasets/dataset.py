# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from utils.process import ProcessUtils as proc
from utils.augment import AugmentUtils as aug
from utils.udaap.utils_augment import Augment


class CommDataset(torch.utils.data.Dataset):
    def __init__(self, dsType, dataArray, means, stds, isAug=False, isDraw=False, **kwargs):
        self.dsType, self.dataArray, self.means, self.stds, self.isAug, self.isDraw = dsType, copy.deepcopy(dataArray), means, stds, isAug, isDraw
        self.inpRes, self.outRes, self.basePath, self.imgType = kwargs['inpRes'], kwargs['outRes'], kwargs['basePath'], kwargs['imgType']
        self.useFlip, self.useOcc, self.sf, self.rf = kwargs['useFlip'], kwargs['useOcclusion'], kwargs['scaleRange'], kwargs['rotRange']
        if self.isAug and self.useOcc:
            self.augmentor = Augment(num_occluder=kwargs['numOccluder'])  # 从voc2012数据集中拿到一些mask形状，来遮挡图像。

    def __getitem__(self, idx):
        # region 1 数据预处理（含图像resize）
        annObj = self.dataArray[idx]
        img, kps, kps_test, imageID = proc.image_load(annObj["imagePath"]), annObj["kps"], annObj["kps_test"], annObj["imageID"]  # H*W*C
        img, kps, _ = proc.image_resize(img, kps, self.inpRes)  # H*W*C
        _, kps_test, _ = proc.image_resize(img, kps_test, self.inpRes)  # 测试用。用于验证选用的伪标签的质量。
        test_kpsMap = torch.from_numpy(np.array(kps_test).astype(np.float32))  # 测试用。用于验证选用的伪标签的质量。
        # endregion

        # region 2. 生成原样本对应数据
        islabeled = torch.tensor(annObj["islabeled"] == 1)
        ori_imgMap = proc.image_np2tensor(img)  # H*W*C ==> C*H*W
        ori_kpsMap = torch.from_numpy(np.array(kps).astype(np.float32))
        ori_heatmap, ori_kpsMap = proc.kps_heatmap(ori_kpsMap, ori_imgMap.shape, self.inpRes, self.outRes)
        ori_kpsWeight = ori_kpsMap[:, 2].clone()
        ori_angle = torch.tensor(0.)
        ori_scale = torch.tensor(self.inpRes / 200.0)
        ori_cenMap = torch.tensor(proc.center_calculate(img))
        ori_isflip = torch.tensor(False)
        # endregion

        # region 3. 增广样本初始化
        aug_imgMap = ori_imgMap.clone()
        aug_kpsMap = ori_kpsMap.clone()
        aug_angle = ori_angle.clone()
        aug_scale = ori_scale.clone()
        aug_cenMap = ori_cenMap.clone()
        aug_isflip = torch.tensor(False)
        # endregion

        # region 4. 数据增强
        if self.isAug:
            # 随机水平翻转
            aug_imgMap, aug_kpsMap, aug_cenMap, aug_isflip = aug.fliplr(aug_imgMap, aug_kpsMap, aug_cenMap, prob=0.5)
            if self.isDraw: draw_imgMap_flip, draw_kpsMap_flip = aug_imgMap.clone(), aug_kpsMap.clone()  # 测试用
            # 随机加噪（随机比例去均值）
            aug_imgMap = aug.noisy_mean(aug_imgMap)
            # 随机仿射变换（随机缩放、随机旋转）
            aug_imgMap, aug_kpsMap, aug_scale, aug_angle = aug.affine(aug_imgMap, aug_kpsMap, aug_cenMap, aug_scale, self.sf, aug_angle, self.rf, [self.inpRes, self.inpRes])
            if self.isDraw: draw_imgMap_affine, draw_kpsMap_affine = aug_imgMap.clone(), aug_kpsMap.clone()  # 测试用
            # 随机遮挡
            if self.useOcc:
                aug_imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(aug_imgMap))
                aug_imgMap = proc.image_np2tensor(aug_imgMap)
                if self.isDraw: draw_imgMap_occlusion = aug_imgMap.clone()  # 测试用
        # endregion

        # region 5.数据处理
        # 图像RGB通道去均值（C*H*W）
        aug_imgMap = proc.image_colorNorm(aug_imgMap, self.means, self.stds)
        # 生成kpsMap对应的heatmap
        aug_heatmap, aug_kpsMap = proc.kps_heatmap(aug_kpsMap, aug_imgMap.shape, self.inpRes, self.outRes)
        aug_kpsWeight = aug_kpsMap[:, 2].clone()
        aug_warpmat = aug.affine_getWarpmat(-aug_angle, 1/aug_scale, matrixRes=[self.inpRes, self.inpRes])
        # endregion

        # region 6.数据增强测试
        if self.isDraw:
            # region 6.1 输出原图（含关键点）
            draw_kpsMap = ori_kpsMap if islabeled else test_kpsMap
            self._draw_testImage(imageID, "01_ori", ori_imgMap.clone(), draw_kpsMap)

            if islabeled:
                draw2_kpsMap = torch.ones((ori_kpsWeight.size(0), 3))
                draw2_kpsMap[:, 0:2] = proc.kps_fromHeatmap(ori_heatmap.detach().cpu(), ori_cenMap, ori_scale, [self.outRes, self.outRes], mode="single")  # 使用ori_scale
                draw2_kpsMap[:, 2] = ori_kpsWeight
                self._draw_testImage(imageID, "02_ori_heatmap", ori_imgMap.clone(), draw2_kpsMap)
            # endregion

            # region 6.2 输出水平翻转图
            if islabeled and self.isAug and self.useFlip:
                self._draw_testImage(imageID, "03_flip", draw_imgMap_flip, draw_kpsMap_flip)
            # endregion

            # region 6.3 输出变换后图（含关键点）
            if islabeled and self.isAug:
                self._draw_testImage(imageID, "04_affine", draw_imgMap_affine, draw_kpsMap_affine)
            # endregion

            # region 6.4 输出遮挡后图（含关键点）
            if self.isAug and self.useOcc:
                self._draw_testImage(imageID, "05_occlus", draw_imgMap_occlusion, draw_kpsMap_affine)
            # endregion

            # region 6.5 输出warpmat变换后图（含关键点）
            if islabeled and self.isAug:
                # region 反向仿射变换
                draw_heatmap = aug_heatmap.clone().unsqueeze(0)
                affine_grid = F.affine_grid(aug_warpmat.unsqueeze(0), draw_heatmap.size(), align_corners=True)
                draw_heatmap = F.grid_sample(draw_heatmap, affine_grid, align_corners=True).squeeze(0)
                # 进行反向水平翻转
                if aug_isflip: draw_heatmap = aug.fliplr_back_tensor(draw_heatmap)
                # 从heatmap中获得关键点
                draw_kpsMap = torch.ones((aug_kpsWeight.size(0), 3))
                draw_kpsMap[:, 0:2] = proc.kps_fromHeatmap(draw_heatmap, aug_cenMap, torch.tensor(1), [self.outRes, self.outRes], mode="single")  # 使用aug_scale
                draw_kpsMap[:, 2] = aug_kpsWeight
                # endregion
                self._draw_testImage(imageID, "06_warpmat", ori_imgMap.clone(), draw_kpsMap)
            # endregion

            # region 6.6 输出最终图（含关键点）-- 只输出标记样本的数据增广样本
            if islabeled and self.isAug:
                draw_kpsMap = torch.ones((aug_kpsWeight.size(0), 3))
                draw_kpsMap[:, 0:2] = proc.kps_fromHeatmap(aug_heatmap.detach().cpu(), aug_cenMap, ori_scale, [self.outRes, self.outRes], mode="single")  # 使用ori_scale
                draw_kpsMap[:, 2] = aug_kpsWeight
                self._draw_testImage(imageID, "07_heatmap", aug_imgMap.clone(), draw_kpsMap)
                self._draw_testImage(imageID, "09_final", aug_imgMap.clone(), aug_kpsMap)
            # endregion

            # region 6.7 输出增广样本的image的反向变换图
            if self.isAug:
                # region 反向仿射变换
                draw_img = aug_imgMap.clone().unsqueeze(0)
                affine_grid = F.affine_grid(aug_warpmat.unsqueeze(0), draw_img.size(), align_corners=True)
                draw_img = F.grid_sample(draw_img, affine_grid, align_corners=True).squeeze(0)
                # 进行反向水平翻转
                if aug_isflip: draw_img = aug.fliplr_back_tensor(draw_img)
                self._draw_testImage(imageID, "09_imgWarpmat", draw_img, [])
            # endregion
        # endregion

        meta = {"imageID": annObj["imageID"], "imagePath": annObj["imagePath"], "islabeled": islabeled,
                "warpmat": aug_warpmat, "kpsMap": aug_kpsMap, 'kpsWeight': aug_kpsWeight, "center": aug_cenMap, "angle": aug_angle, "scale": aug_scale, "isflip": aug_isflip,
                "ori_imgMap": ori_imgMap, "ori_kpsMap": ori_kpsMap, 'ori_kpsWeight': ori_kpsWeight, "ori_cenMap": ori_cenMap, "ori_angle": ori_angle, "ori_scale": ori_scale, "ori_isflip": ori_isflip,
                "kpsMap_test": test_kpsMap}
        return aug_imgMap, aug_heatmap, meta

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
        del img_draw

