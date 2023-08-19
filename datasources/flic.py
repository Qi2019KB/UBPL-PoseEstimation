# -*- coding: utf-8 -*-
import GLOB as glob
import datetime
import copy
import math
import cv2
import scipy.io as io
from tqdm import tqdm
import numpy as np
import random
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc


# 参考：https://zhuanlan.zhihu.com/p/225494133
class FLICData:
    def __init__(self):
        self.imgPath = "D:/00Data/pose/FLIC/images"
        self.labelPathname = "D:/00Data/pose/FLIC/examples.mat"
        self.imgType = "jpg"
        self.inpRes = 256
        self.outRes = 64
        self.pck_ref = [3, 7]  # 左肩、左腰 ==> pck_thr推荐0.2
        self.pck_thr = 0.5
        self.selKpIdxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.kpsCount = len(self.selKpIdxs)

    # 获得全标签数据（用于监督学习）
    def getData(self, trainCount, validCount, reMean=True):
        candiArray = copy.deepcopy(self._dataLoading())
        random.shuffle(candiArray)
        trainData, validData = candiArray[0: trainCount], candiArray[trainCount: trainCount + validCount]
        [trainData, validData] = self._data_cache([trainData, validData], [trainCount, validCount])
        if reMean:
            means, stds = self._getNormParams(trainData + validData)
        else:
            means, stds = [0.25195965, 0.22432944, 0.20951675], [0.23108867, 0.22090606, 0.22124061]
        return trainData, validData, means, stds

    # 获得半标签数据（用于半监督学习）
    def getSemiData(self, trainCount, validCount, labelRatio, reMean=True):
        candiArray = copy.deepcopy(self._dataLoading())
        random.shuffle(candiArray)
        trainData, validData = candiArray[0: trainCount], candiArray[trainCount: trainCount + validCount]
        semiTrainData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs = self._semiOrgan(trainData, labelRatio)
        [semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs] = self._data_cache([semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs], [trainCount, validCount, labelRatio])
        if reMean:
            means, stds = self._getNormParams(trainData + validData)
        else:
            means, stds = [0.25195965, 0.22432944, 0.20951675], [0.23108867, 0.22090606, 0.22124061]
        return semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs, means, stds

    # 数据加载
    def _dataLoading(self):
        matFile = io.loadmat(self.labelPathname)['examples']
        labelArray, imgNameArray, kpsArray = [], matFile['filepath'][0], matFile['coords'][0]

        for imgIdx, imgName in enumerate(imgNameArray):
            imgName, kps = imgName[0], [[int(kp[0]), int(kp[1]), 1] for kp in kpsArray[imgIdx].T if math.isnan(kp[0]) == False]
            # 过滤掉关键点不全的图片
            kps_new = []
            for kpIdx, kp in enumerate(kps):
                if kpIdx in self.selKpIdxs and kp[2] > 0: kps_new.append([kp[0], kp[1], 1])
            if len(kps_new) < self.kpsCount: continue
            id = "im{}".format(str(1000000 + imgIdx + 1)[3:])
            imgID = comm.file_splitFilename(imgName)
            # label组织
            labelArray.append({
                "islabeled": 1,
                "id": id,
                "imageID": imgID,
                "imageName": imgName,
                "imagePath": "{}/{}".format(self.imgPath, imgName),
                "kps": kps_new,
                "kps_test": kps_new
            })
        # region 过滤掉多人的样本
        labelArray_new = []
        for cItem in labelArray:
            count = len([item for item in labelArray if item["imageID"] == cItem["imageID"]])
            if count == 1:
                labelArray_new.append(cItem)
        # endregion
        return labelArray_new

    # 计算数据均值和方差
    def _getNormParams(self, labelArray):
        means, stds, imgArray = [], [], []
        imgPaths = [annObj["imagePath"] for annObj in labelArray]
        for imgPath in tqdm(imgPaths):
            img = cv2.imread(imgPath)
            img = cv2.resize(img, (self.inpRes, self.inpRes))
            img = img[:, :, :, np.newaxis]
            imgArray.append(img)
        imgs = np.concatenate(imgArray, axis=3)
        imgs = imgs.astype(np.float32) / 255.
        for i in range(3):
            pixels = imgs[:, :, i, :].ravel()  # 拉成一行
            means.append(np.mean(pixels))
            stds.append(np.std(pixels))
        # BGR --> RGB ， CV读取的需要转换（这里需要），PIL读取的不用转换
        means.reverse()
        stds.reverse()
        return means, stds

    # 半监督数据组织
    def _semiOrgan(self, trainData, labeledRatio):
        labeledCount = int(len(trainData) * labeledRatio)
        unlabeledCount = len(trainData) - labeledCount

        semiDataArray, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs = [], [], [], [], []
        voidIdxs = random.sample([item for item in range(0, len(trainData))], unlabeledCount)
        for idx, item in enumerate(trainData):
            if idx in voidIdxs:
                dataItem = copy.deepcopy(item)
                dataItem["islabeled"] = 0
                dataItem["kps"] = [[0, 0, 0] for i in range(self.kpsCount)]
                unlabeledIdxs.append(idx)
                unlabeledData.append(dataItem)
                semiDataArray.append(dataItem)
            else:
                dataItem = copy.deepcopy(item)
                dataItem["islabeled"] = 1
                labeledIdxs.append(idx)
                labeledData.append(dataItem)
                semiDataArray.append(dataItem)
        return semiDataArray, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs

    def _data_cache(self, dataArray, paramArray):
        saveName = "FLIC"
        for item in paramArray:
            saveName += "_{}".format(item)
        savePath = "{}/datasources/temp_data/{}.json".format(glob.root, saveName)
        if not comm.file_isfile(savePath):
            comm.json_save(dataArray, savePath, isCover=True)
            return dataArray
        else:
            return comm.json_load(savePath)


if __name__ == "__main__":
    basePath = "{}/dataset/flic_{}".format(glob.temp, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    dataSource = FLICData()
    trainData, validData, means, stds = dataSource.getData(10, 10, reMean=False)  # 3987, 1016
    for annObj in tqdm(trainData):
        img = proc.image_load(annObj["imagePath"])
        for kIdx, kp in enumerate(annObj["kps"]):
            color = (255, 0, 0) if kIdx in dataSource.pck_ref else (0, 95, 191)
            img = proc.draw_point(img, kp, color, text="k{}".format(kIdx), textScale=0.7, textColor=color)
        savePath = "{}/{}".format(basePath, annObj["imageName"])
        proc.image_save(img, savePath)
