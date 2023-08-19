# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import GLOB as glob
import datetime
import copy
import cv2
from tqdm import tqdm
import numpy as np
import random
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc

import torch
import torchvision


class CIFAR10Data:
    def __init__(self):
        self.imgType = "png"
        self.inpRes = 32
        self.outRes = 32
        self.train_dataArray, self.valid_dataArray, self.train_labelArray, self.valid_labelArray, self.classes, self.num_classes = self._dataLoading("D:/00Data/cifar10(Classification)/data")

    # 获得全标签数据（用于监督学习）
    def getData(self, trainCount, validCount):
        train_candiArray = copy.deepcopy(self.train_labelArray)
        random.shuffle(train_candiArray)
        trainData = train_candiArray[0:trainCount]

        valid_candiArray = copy.deepcopy(self.valid_labelArray)
        random.shuffle(valid_candiArray)
        validData = valid_candiArray[0:validCount]

        [trainData, validData] = self._data_cache([trainData, validData], [trainCount, validCount])
        means, stds = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        return trainData, validData, means, stds

    # 获得半标签数据（用于半监督学习）
    def getSemiData(self, trainCount, validCount, labelRatio):
        train_candiArray = copy.deepcopy(self.train_labelArray)
        random.shuffle(train_candiArray)
        trainData = train_candiArray[0:trainCount]

        valid_candiArray = copy.deepcopy(self.valid_labelArray)
        random.shuffle(valid_candiArray)
        validData = valid_candiArray[0:validCount]

        semiTrainData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs = self._semiOrgan(trainData, labelRatio)
        [semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs] = self._data_cache([semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs], [trainCount, validCount, labelRatio])
        means, stds = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        return semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs, means, stds

    # 数据加载
    def _dataLoading(self, dataPath):
        train_dataArray, train_labelArray, classes, num_classes = self._get_C10Data(dataPath, True)
        valid_dataArray, valid_labelArray, _, _ = self._get_C10Data(dataPath, False)
        return train_dataArray, valid_dataArray, train_labelArray, valid_labelArray, classes, num_classes

    def _get_C10Data(self, dataPath, isTrain):
        dataset = torchvision.datasets.CIFAR10(root=dataPath, train=isTrain, download=True)
        dataArray, targets = dataset.data, dataset.targets

        labelArray = []
        id_start = 1100000 if isTrain else 1200000
        for dIdx, dataItem in enumerate(dataArray):
            id = "im{}".format(str(id_start + dIdx + 1)[1:])
            # label组织
            labelArray.append({
                "islabeled": 1,
                "id": id,
                "imageID": id,
                "imageName": "{}.{}".format(id, self.imgType),
                "label": targets[dIdx],
                "label_test": targets[dIdx]
            })
        return dataArray, labelArray, dataset.classes, len(dataset.classes)

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
                dataItem["label"] = -1
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
        saveName = "cifar10"
        for item in paramArray:
            saveName += "_{}".format(item)
        savePath = "{}/datasources/temp_data/{}.json".format(glob.root, saveName)
        if not comm.file_isfile(savePath):
            comm.json_save(dataArray, savePath, isCover=True)
            return dataArray
        else:
            return comm.json_load(savePath)


if __name__ == "__main__":
    basePath = "{}/dataset/cifar10_{}".format(glob.temp, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    dataSource = CIFAR10Data()
    trainData, validData, means, stds = dataSource.getData(20, 200)  # 1000, 200
    for annObj in tqdm(trainData):
        imgIdx = int(annObj["imageID"][3:])-1
        img = dataSource.train_dataArray[imgIdx]
        savePath = "{}/{}".format(basePath, annObj["imageName"])
        proc.image_save(img, savePath)

    for annObj in tqdm(validData):
        imgIdx = int(annObj["imageID"][3:])-1
        img = dataSource.valid_dataArray[imgIdx]
        savePath = "{}/{}".format(basePath, annObj["imageName"])
        proc.image_save(img, savePath)
