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
    def __init__(self, dsType, imageArray, dataArray, means, stds, isAug=False, isDraw=False, **kwargs):
        self.imageArray = imageArray
        self.dsType, self.dataArray, self.means, self.stds, self.isAug, self.isDraw = dsType, copy.deepcopy(dataArray), means, stds, isAug, isDraw
        self.inpRes, self.outRes, self.basePath, self.imgType = kwargs['inpRes'], kwargs['outRes'], kwargs['basePath'], kwargs['imgType']
        self.useFlip, self.useOcc, self.sf, self.rf = kwargs['useFlip'], kwargs['useOcclusion'], kwargs['scaleRange'], kwargs['rotRange']
        if self.isAug and self.useOcc:
            self.augmentor = Augment(num_occluder=kwargs['numOccluder'])  # 从voc2012数据集中拿到一些mask形状，来遮挡图像。

    def __getitem__(self, idx):
        # region 1 数据预处理（含图像resize）
        annObj = self.dataArray[idx]
        img, label, test_label, imageID = self.imageArray[int(annObj["imageID"][3:])-1], annObj["label"], annObj["label_test"], annObj["imageID"]  # H*W*C
        # endregion

        # region 2. 生成原样本对应数据
        islabeled = torch.tensor(annObj["islabeled"] == 1)
        ori_imgMap = proc.image_np2tensor(img)  # H*W*C ==> C*H*W
        ori_angle = torch.tensor(0.)
        ori_scale = torch.tensor(self.inpRes / 200.0)  # 待修改 0413
        ori_cenMap = torch.tensor(proc.center_calculate(img))
        ori_isflip = torch.tensor(False)
        # endregion

        # region 3. 增广样本初始化
        aug_imgMap = ori_imgMap.clone()
        aug_angle = ori_angle.clone()
        aug_scale = ori_scale.clone()
        aug_cenMap = ori_cenMap.clone()
        aug_isflip = torch.tensor(False)
        # endregion

        # region 4. 数据增强
        if self.isAug:
            # 随机水平翻转
            aug_imgMap, aug_cenMap, aug_isflip = aug.fliplr_classification(aug_imgMap, aug_cenMap, prob=0.5)
            if self.isDraw: draw_imgMap_flip = aug_imgMap.clone()  # 测试用
            # 随机加噪（随机比例去均值）
            aug_imgMap = aug.noisy_mean(aug_imgMap)
            # 随机仿射变换（随机缩放、随机旋转）
            aug_imgMap, aug_scale, aug_angle = aug.affine_classification(aug_imgMap, aug_cenMap, aug_scale, self.sf, aug_angle, self.rf, [self.inpRes, self.inpRes])
            if self.isDraw: draw_imgMap_affine = aug_imgMap.clone()  # 测试用
            # 随机遮挡
            if self.useOcc:
                aug_imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(aug_imgMap))
                aug_imgMap = proc.image_np2tensor(aug_imgMap)
                if self.isDraw: draw_imgMap_occlusion = aug_imgMap.clone()  # 测试用
        # endregion

        # region 5.数据处理
        # 图像RGB通道去均值（C*H*W）
        aug_imgMap = proc.image_colorNorm(aug_imgMap, self.means, self.stds)
        aug_warpmat = aug.affine_getWarpmat(-aug_angle, 1, matrixRes=[self.inpRes, self.inpRes])
        # endregion

        # region 6.数据增强测试
        if self.isDraw:
            # region 6.1 输出原图（含关键点）
            self._draw_testImage(imageID, "01_ori", ori_imgMap.clone())
            # endregion

            # region 6.2 输出水平翻转图
            if self.isAug and self.useFlip:
                self._draw_testImage(imageID, "03_flip", draw_imgMap_flip)
            # endregion

            # region 6.3 输出变换后图（含关键点）
            if self.isAug:
                self._draw_testImage(imageID, "04_affine", draw_imgMap_affine)
            # endregion

            # region 6.4 输出遮挡后图（含关键点）
            if self.isAug and self.useOcc:
                self._draw_testImage(imageID, "05_occlus", draw_imgMap_occlusion)
            # endregion

            # region 6.5 输出warpmat变换后图（含关键点）
            if self.isAug:
                # region 反向仿射变换
                draw_img = aug_imgMap.clone().unsqueeze(0)
                affine_grid = F.affine_grid(aug_warpmat.unsqueeze(0), draw_img.size(), align_corners=True)
                draw_img = F.grid_sample(draw_img, affine_grid, align_corners=True).squeeze(0)
                # 进行反向水平翻转
                if aug_isflip: draw_img = aug.fliplr_back_tensor(draw_img)
                # endregion
                self._draw_testImage(imageID, "06_warpmat", draw_img)
            # endregion
        # endregion

        meta = {"imageID": annObj["imageID"], "islabeled": islabeled,
                "warpmat": aug_warpmat, "center": aug_cenMap, "angle": aug_angle, "scale": aug_scale, "isflip": aug_isflip,
                "ori_imgMap": ori_imgMap, "ori_cenMap": ori_cenMap, "ori_angle": ori_angle, "ori_scale": ori_scale, "ori_isflip": ori_isflip,
                "label_test": test_label}
        return aug_imgMap, label, meta

    def __len__(self):
        return len(self.dataArray)

    def _draw_testImage(self, imageID, stepID, imgMap):
        img_draw = proc.image_tensor2np(imgMap.detach().cpu().data * 255).astype(np.uint8)
        proc.image_save(img_draw, "{}/draw/dataset/{}/{}_{}.{}".format(self.basePath, self.dsType, imageID, stepID, self.imgType))
        del img_draw

