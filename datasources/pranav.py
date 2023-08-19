# -*- coding: utf-8 -*-
import GLOB as glob
import datetime
import copy
import cv2
from tqdm import tqdm
import numpy as np
import random
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc


class PranavData:
    def __init__(self, inpRes=256):
        self.labelPathname = "D:/00Data/pose/Openfield-Pranav/box_train/data.json"
        self.imgPath = "D:/00Data/pose/Openfield-Pranav/box_train/img"
        self.imgType = "png"
        self.inpRes = inpRes
        self.kpsCount = 4
        self.pck_ref = [1, 2]  # 左右耳

    # 获得全标签数据（用于监督学习）
    def getData(self, trainCount, validCount, reMean=True):
        candiArray = copy.deepcopy(self._dataLoading())
        random.shuffle(candiArray)
        trainData, validData = candiArray[0: trainCount], candiArray[trainCount: trainCount + validCount]
        [trainData, validData] = self._data_cache([trainData, validData], [trainCount, validCount])
        if reMean:
            means, stds = self._getNormParams(trainData + validData)
        else:
            means, stds = [0.4920829, 0.4920829, 0.4920829], [0.16629942, 0.16629942, 0.16629942]
        return trainData, validData, means, stds

    # 获得半标签数据（用于半监督学习）
    def getSemiData(self, trainCount, validCount, labelRatio, reMean=True):
        candiArray = copy.deepcopy(self._dataLoading())
        random.shuffle(candiArray)
        trainData, validData = candiArray[0: trainCount], candiArray[trainCount: trainCount + validCount]
        semiTrainData, labeledData, unlabeledData, _, _ = self._semiOrgan(trainData, labelRatio)
        [semiTrainData, validData, labeledData, unlabeledData] = self._data_cache([semiTrainData, validData, labeledData, unlabeledData], [trainCount, validCount, labelRatio])
        if reMean:
            means, stds = self._getNormParams(trainData + validData)
        else:
            means, stds = [0.4920829, 0.4920829, 0.4920829], [0.16629942, 0.16629942, 0.16629942]
        return semiTrainData, validData, labeledData, unlabeledData, means, stds

    # 数据加载
    def _dataLoading(self):
        imgPathArray = comm.file_scan("{}/*.{}".format(self.imgPath, self.imgType))
        kpsArray = comm.json_load(self.labelPathname)
        labelArray = []
        for imgIdx, imgPath in enumerate(imgPathArray):
            id = "im{}".format(str(1000000 + imgIdx + 1)[3:])
            imgID = comm.file_splitFilename(imgPath)
            imgName = "{}.{}".format(imgID, self.imgType)

            kps = [[int(kp[0]), int(kp[1]), 1] for kp in kpsArray[imgName]]
            # label组织
            labelArray.append({
                "islabeled": 1,
                "id": id,
                "imageID": imgID,
                "imageName": imgName,
                "imagePath": "{}/{}".format(self.imgPath, imgName),
                "kps": kps,
                "kps_test": kps
            })
        return labelArray

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
        saveName = "PRANAV"
        for item in paramArray:
            saveName += "_{}".format(item)
        savePath = "{}/datasources/temp_data/{}.json".format(glob.root, saveName)
        if not comm.file_isfile(savePath):
            comm.json_save(dataArray, savePath, isCover=True)
            return dataArray
        else:
            return comm.json_load(savePath)


if __name__ == "__main__":
    basePath = "{}/dataset/pranav_{}".format(glob.temp, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    dataSource = PranavData()
    trainData, validData, means, stds = dataSource.getData(20, 200, reMean=False)  # 1000, 200
    for annObj in tqdm(trainData):
        img = proc.image_load(annObj["imagePath"])
        for kIdx, kp in enumerate(annObj["kps"]):
            color = (255, 0, 0) if kIdx in dataSource.pck_ref else (0, 95, 191)
            img = proc.draw_point(img, kp, color, text="k{}".format(kIdx+1), textScale=0.4, textColor=color)
        savePath = "{}/{}".format(basePath, annObj["imageName"])
        proc.image_save(img, savePath)
