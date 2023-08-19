# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from utils.base.comm import CommUtils as comm
from utils.augment import AugmentUtils as aug


class FeaturePool(nn.Module):
    def __init__(self, maxLength):
        super(FeaturePool, self).__init__()
        self.pool = []
        self.maxLength = maxLength

    def save_features(self, imageIDs, features, warpmat, isFlip):  # features: [bs, 128, 1, 1]
        features_ab = aug.affine_back2(features, warpmat, isFlip).detach().cpu()
        for bsIdx in range(features.shape[0]):
            imageID, feature_ab = imageIDs[bsIdx], features_ab[bsIdx].detach()
            sample = self._find_sample("imageID", imageID)
            if sample is None:
                sample = {"imageID": imageID, "feature": [feature_ab]}
                self.pool.append(sample)
            else:
                sample["feature"].append(feature_ab)
            if len(sample["feature"]) >= self.maxLength:
                sample["feature"] = sample["feature"][(len(sample["feature"]) - self.maxLength):]

    def get_features_mixture(self, imageIDs, epo):
        feature_mixtures = []
        for imageID in imageIDs:
            sample = self._find_sample("imageID", imageID)
            epoes = [idx for idx in range((epo - len(sample["feature"]) + 1), (epo + 1))]
            ws = [self._feature_mixture_weight(epoIdx) for epoIdx in epoes]
            val = torch.zeros_like(sample["feature"][-1])
            for wIdx in range(len(ws)):
                val += ws[wIdx] * sample["feature"][-(wIdx+1)]
            feature_mixture = val/sum(ws)
            feature_mixtures.append(feature_mixture)
        feature_mixtures = torch.stack(feature_mixtures, dim=0)
        return feature_mixtures

    def _feature_mixture_weight(self, epo, l=10):
        t = epo - l
        return (1 - comm.math_signal(t) + comm.math_signal(t)*math.exp(-comm.math_signal(t)*t))/2

    def _find_sample(self, key, value):
        instance = None
        for item in self.pool:
            if item[key] == value:
                instance = item
                break
        return instance


def get_feature_pools(pool_maxLength, modelNum=1, augNum=1, nStack=1):
    feature_pools = []
    for mIdx in range(modelNum):
        pool_aug = []
        for aIdx in range(augNum):
            pool_stack = []
            for nIdx in range(nStack):
                pool_stack.append(FeaturePool(pool_maxLength))
            pool_aug.append(pool_stack)
        feature_pools.append(pool_aug)
    return feature_pools