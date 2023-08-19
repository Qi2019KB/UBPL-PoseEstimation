# -*- coding: utf-8 -*-
import math
import copy
import numpy as np
import torch
from utils.process import ProcessUtils as proc
from utils.evaluation import EvaluationUtils as eval


class BusinessUtils:
    def __init__(self):
        pass

    # Assess the quality of predictions
    @classmethod
    def assess_pseudo_unc(cls, imageIDs, test_kpsMap, preds, args):
        norms = eval.acc_pck_pseudo_norm(imageIDs, test_kpsMap, args.pck_ref)
        predNum, [bs, k, _] = len(preds), preds[0].shape

        pseudoArray = []
        for pIdx in range(predNum):
            pseudoes = []
            for bsIdx in range(bs):
                imageID = imageIDs[bsIdx]
                norm = [item["norm"] for item in norms if item["imageID"] == imageID][0]
                for kIdx in range(k):
                    kID = "{}_{}".format(imageID, kIdx)
                    kCoord = preds[pIdx][bsIdx, kIdx].cpu().data.numpy().tolist()
                    kGT = test_kpsMap[bsIdx, kIdx].cpu().data.numpy().tolist()
                    kError, kAccFlag = cls._check_predsQuality(cls, kCoord, kGT, norm, args)
                    kLegal = 1.0 if kCoord[0] >= 0 and kCoord[1] >= 0 else 0.
                    pseudoes.append({"kpID": kID, "imageID": imageID, "kIdx": kIdx, "coord": kCoord, "coord_gt": kGT,
                                     "coord_legal": kLegal, "error": kError, "acc_flag": kAccFlag})
            pseudoArray.append(pseudoes)
        return pseudoArray

    def _check_predsQuality(self, coord, gt, norm, args):
        err = proc.coord_distance(coord, gt)
        accFlag = eval.acc_pck_pseudo(err, norm, args.pck_thr)
        return err, accFlag


    def _calReliabilityThr(self, dataArray, args):
        scores = [item["reliability"] for item in dataArray]
        scoreThr = max(args.reliableThr, scores[int((len(scores) - 1) * args.reliablePCT)])
        return scoreThr

    @classmethod
    def filter_pseudo(cls, predsArraies, args):
        predsArray_mds1, predsArray_mds2, pseudoArray = predsArraies
        # region 1. Assess the quality of ensemble pseudo-labels
        dists = []
        for pIdx, pseudo in enumerate(pseudoArray):
            kSample_mds1, kSample_mds2 = predsArray_mds1[pIdx], predsArray_mds2[pIdx]
            dist = proc.coord_distance(kSample_mds1["coord"], kSample_mds2["coord"])
            dists.append(dist)
            pseudo["dist"] = dist
            pseudo["coord_legal"] = kSample_mds1["coord_legal"] and kSample_mds2["coord_legal"]
        dist_max, dist_min = np.max(dists), np.min(dists)
        if dist_min > args.reliableDistMin: dist_min = args.reliableDistMin
        # cal the reliability score
        for pIdx, pseudo in enumerate(pseudoArray):
            unc_score = (pseudo["dist"]-dist_min)/(dist_max-dist_min) if (pseudo["coord_legal"] > 0.) else 1.0
            pseudo["reliability"] = 1.0 - unc_score
        pseudoArray = sorted(pseudoArray, key=lambda x: x["reliability"], reverse=True)
        # endregion

        # region 3. Pick pseudo-labels
        reliableThr = cls._calReliabilityThr(cls, pseudoArray, args)
        # pseudo-label filter
        selArray, selCounts, selErrs, selAccs = [], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)]
        for pseudoItem in pseudoArray:
            item = copy.deepcopy(pseudoItem)
            if item["reliability"] > reliableThr:
                kID = int(item["kpID"].split("_")[-1])
                item["enable"] = 1
                selCounts[-1] += 1
                selCounts[kID] += 1
                selErrs[-1] += item["error"]
                selErrs[kID] += item["error"]
                selAccs[-1] += item["acc_flag"]
                selAccs[kID] += item["acc_flag"]
            else:
                item["enable"] = 0
            selArray.append(item)
        for idx in range(args.kpsCount+1):
            if selCounts[idx] > 0:
                selErrs[idx] = selErrs[idx] / selCounts[idx]
                selAccs[idx] = selAccs[idx] / selCounts[idx]
        return selArray, selCounts, selErrs, selAccs, reliableThr
        # endregion

    # region function：伪标签质量评价
    # idea:
    #   （1）intDist_mean: ① 每个teacher对N个augSamples的预测距离的均值，评价该模型对该样本的认知程度。
    #                     ② 对两个teacher的intDist_mean做归一化后，对集成预测集成加权均值。
    #   （2）extDist_mean: 两个teacher对每个augSample的预测距离的均值，评价该伪标签的质量
    # params:
    #   （1）imageIDs：样本ID
    #   （2）test_kpsMap：真值（不用于模型的训练，仅用于评价伪标签挑选的挑选质量）
    #   （3）ori_predsArray：模型对原样本的预测
    #   （4）augs_predsArray：模型对增广样本的预测
    #   （5）args：运行参数
    # return:
    #   （1）ori_assessArraies：原样本预测质量评价
    #   （2）augs_assessArraies：增广样本预测质量评价
    # endregion
    @classmethod
    def assess_pseudo_unc2(cls, imageIDs, test_kpsMap, ori_predsArray, augs_predsArray, args):
        # region 1. 评价各模型对各样本的预测质量
        ori_assessArraies = cls.assess_pseudo_unc(imageIDs, test_kpsMap, ori_predsArray, args)
        augs_assessArraies = [cls.assess_pseudo_unc(imageIDs, test_kpsMap, aug_predsArray, args) for aug_predsArray in augs_predsArray]
        # endregion

        # region 2. 评价伪标签质量
        pseudoArray = []
        for iIdx in range(len(ori_assessArraies[-1])):
            # region 2.1 数据准备
            pseudo = copy.deepcopy(ori_assessArraies[-1][iIdx])
            pseudo["coord_w1"], pseudo["coord_w2"] = 0.5, 0.5
            pseudo["intDist1"], pseudo["intDist2"], pseudo["extDist"] = 999, 999, 999
            oriArray = cls._getAssessesByIdx(cls, ori_assessArraies, iIdx)
            mds1_augArray = cls._getAssessesByIdx(cls, augs_assessArraies[0], iIdx)
            mds2_augArray = cls._getAssessesByIdx(cls, augs_assessArraies[1], iIdx)
            # endregion

            ori_legal = oriArray[0]["coord_legal"] > 0 and oriArray[1]["coord_legal"] > 0
            pseudo["coord_legal"] = 1.0 if ori_legal else 0.0
            if ori_legal and cls._checkGroupLegal(cls, mds1_augArray) and cls._checkGroupLegal(cls, mds2_augArray):
                # region 2.2 计算intDist（每个teacher对N个augSamples的预测距离的均值），评价该模型对该样本的认知程度。并以加权平均方式计算集成预测。
                mds1_intDist = proc.coord_avgDistance([item["coord"] for item in mds1_augArray])
                mds2_intDist = proc.coord_avgDistance([item["coord"] for item in mds2_augArray])
                w1, p1 = mds1_intDist/(mds1_intDist+mds2_intDist), oriArray[0]["coord"]
                w2, p2 = mds2_intDist/(mds1_intDist+mds2_intDist), oriArray[1]["coord"]
                pseudo["coord"] = [(w1*p1[0]+w2*p2[0]), (w1*p1[1]+w2*p2[1])]
                pseudo["coord_legal"] = 1.0
                pseudo["intDist1"] = mds1_intDist
                pseudo["intDist2"] = mds2_intDist
                pseudo["coord_w1"] = w1
                pseudo["coord_w2"] = w2
                # endregion

                # region 2.3 计算extDist（两个teacher对每个augSample的预测距离的均值），评价该伪标签的质量
                dist_sum, dist_n = 0., 0
                for aIdx in range(args.br_inferAugNum):
                    mds1_aug, mds2_aug = mds1_augArray[aIdx], mds2_augArray[aIdx]
                    dist_sum += proc.coord_distance(mds1_aug["coord"], mds2_aug["coord"])
                    dist_n += 1
                pseudo["extDist"] = dist_sum/dist_n
                # endregion
            pseudoArray.append(pseudo)

        # region 2.4 计算伪标签误差
        norms = eval.acc_pck_pseudo_norm(imageIDs, test_kpsMap, args.pck_ref)
        for pseudo in pseudoArray:
            norm = [item["norm"] for item in norms if item["imageID"] == pseudo["imageID"]][0]
            pseudo["error"], pseudo["acc_flag"] = cls._check_predsQuality(cls, pseudo["coord"], pseudo["coord_gt"], norm, args)
        # endregion
        # endregion

        return pseudoArray, ori_assessArraies, augs_assessArraies

    def _getAssessesByIdx(self, assessArraies, idx):
        return [assessArray[idx] for assessArray in assessArraies]

    def _checkGroupLegal(self, itemArray):
        unlegalNum = 0
        for item in itemArray:
            if item["coord_legal"] == 0: unlegalNum += 1
        return unlegalNum == 0

    @classmethod
    def filter_pseudo2(cls, pseudoArray, args):
        # region 1. Assess the reliability of ensemble pseudo-labels
        # region 1.1 修正extDist。使用次高修正999
        dist_max, dist_min = 0, 999
        for pseudo in pseudoArray:
            extDist = pseudo["extDist"]
            if dist_max < extDist < 999: dist_max = extDist
            if dist_min > extDist: dist_min = extDist
        if dist_max == 0: dist_max = 999
        if dist_min > args.reliableDistMin: dist_min = args.reliableDistMin
        # endregion

        # region 1.2 Cal the reliability score
        for pseudo in pseudoArray:
            extDist = pseudo["extDist"] if pseudo["extDist"] != 999 else dist_max
            unc_score = (extDist-dist_min)/(dist_max-dist_min) if (pseudo["coord_legal"] > 0.) else 1.0
            pseudo["reliability"] = 1.0 - unc_score
        pseudoArray = sorted(pseudoArray, key=lambda x: x["reliability"], reverse=True)
        # endregion
        # endregion

        # region 2. Pick pseudo-labels
        reliableThr = cls._calReliabilityThr(cls, pseudoArray, args)
        # pseudo-label filter
        selArray, selCounts, selErrs, selAccs = [], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)]
        for pseudoItem in pseudoArray:
            item = copy.deepcopy(pseudoItem)
            if item["reliability"] > reliableThr:
                kID = int(item["kpID"].split("_")[-1])
                item["enable"] = 1
                selCounts[-1] += 1
                selCounts[kID] += 1
                selErrs[-1] += item["error"]
                selErrs[kID] += item["error"]
                selAccs[-1] += item["acc_flag"]
                selAccs[kID] += item["acc_flag"]
            else:
                item["enable"] = 0
            selArray.append(item)
        for idx in range(args.kpsCount+1):
            if selCounts[idx] > 0:
                selErrs[idx] = selErrs[idx] / selCounts[idx]
                selAccs[idx] = selAccs[idx] / selCounts[idx]
        # endregion
        return selArray, selCounts, selErrs, selAccs, reliableThr

    @classmethod
    def pseudo_cal_unc(cls, imageIDs, preds_gt, preds_mds1, scores_mds1, augPredsArray_mds1, augScoresArray_mds1, preds_mds2, scores_mds2, augPredsArray_mds2, augScoresArray_mds2, args):
        norms = eval.acc_pck_pseudo_norm(imageIDs, preds_gt, args.pck_ref)
        bsNum, kNum = augPredsArray_mds1.size(0), augPredsArray_mds1.size(1)
        pseudoArray_mds1, pseudoArray_mds2 = [], []
        for bsIdx in range(bsNum):
            for kIdx in range(kNum):
                # init mds1_kSample
                mds1_kSample = cls._initKSample(cls, bsIdx, kIdx, imageIDs[bsIdx], preds_gt, norms, preds_mds1, scores_mds1, augPredsArray_mds1, augScoresArray_mds1, args)
                # init mds2_kSample
                mds2_kSample = cls._initKSample(cls, bsIdx, kIdx, imageIDs[bsIdx], preds_gt, norms, preds_mds2, scores_mds2, augPredsArray_mds2, augScoresArray_mds2, args)
                pseudoArray_mds1.append(mds1_kSample)
                pseudoArray_mds2.append(mds2_kSample)
        for idx in range(len(pseudoArray_mds1)):
            pseudoArray_mds1[idx], pseudoArray_mds2[idx] = cls._calKSampleExterData(cls, pseudoArray_mds1[idx], pseudoArray_mds2[idx], args)
        return pseudoArray_mds1, pseudoArray_mds2

    @classmethod
    def pseudo_filter_mixUnc(self, pseudoArray, args):
        # scoreThr = self._calScoreThr(self, pseudoArray, args)
        # for pseudoItem in pseudoArray:
        #     pseudoItem["scoreOK"] = 0 if pseudoItem["score"] <= scoreThr else 1
        #     pseudoItem["unc"] = 999.0 if pseudoItem["score"] <= scoreThr else pseudoItem["unc"]
        uncThr = self._calUncValue(self, args.distThrMax*3)
        # pseudo-label filter
        selArray, selCounts, selErrs, selAccs = [], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)]
        for pseudoItem in pseudoArray:
            item = copy.deepcopy(pseudoItem)
            if item["unc"] <= uncThr:
                kID = int(item["kpID"].split("_")[-1])
                item["enable"] = 1
                selCounts[-1] += 1
                selCounts[kID] += 1
                selErrs[-1] += item["error"]
                selErrs[kID] += item["error"]
                selAccs[-1] += item["acc_flag"]
                selAccs[kID] += item["acc_flag"]
            else:
                item["enable"] = 0
            selArray.append(item)
        for idx in range(args.kpsCount+1):
            if selCounts[idx] > 0:
                selErrs[idx] = selErrs[idx] / selCounts[idx]
                selAccs[idx] = selAccs[idx] / selCounts[idx]
        # endregion
        return selArray, selCounts, selErrs, selAccs, uncThr

    @classmethod
    def pseudo_filter_mixUnc2(self, pseudoArray, args):
        scoreThr = self._calScoreThr(self, pseudoArray, args)
        for pseudoItem in pseudoArray:
            pseudoItem["scoreOK"] = 0 if pseudoItem["score"] < scoreThr else 1
            pseudoItem["unc"] = 999.0 if pseudoItem["score"] < scoreThr else pseudoItem["unc"]
        uncThr = self._calUncValue(self, args.distThrMax*3)
        # pseudo-label filter
        selArray, selCounts, selErrs, selAccs = [], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)]
        for pseudoItem in pseudoArray:
            item = copy.deepcopy(pseudoItem)
            if item["unc"] <= uncThr:
                kID = int(item["kpID"].split("_")[-1])
                item["enable"] = 1
                selCounts[-1] += 1
                selCounts[kID] += 1
                selErrs[-1] += item["error"]
                selErrs[kID] += item["error"]
                selAccs[-1] += item["acc_flag"]
                selAccs[kID] += item["acc_flag"]
            else:
                item["enable"] = 0
            selArray.append(item)
        for idx in range(args.kpsCount+1):
            if selCounts[idx] > 0:
                selErrs[idx] = selErrs[idx] / selCounts[idx]
                selAccs[idx] = selAccs[idx] / selCounts[idx]
        # endregion
        return selArray, selCounts, selErrs, selAccs, scoreThr, uncThr

    @classmethod
    def preds_mean(cls, preds1, preds2):
        preds_mix = torch.stack([preds1, preds2], dim=-1)
        preds_mean = torch.mean(preds_mix, dim=-1)
        return preds_mean

    def _initKSample(self, bsIdx, kIdx, imageID, preds_gt, norms, preds_mds1, scores_mds1, augPredsArray_mds1, augScoresArray_mds1, args):
        k_id = "{}_{}".format(imageID, kIdx)
        k_coord, k_gt = preds_mds1[bsIdx][kIdx].cpu().data.numpy().tolist(), preds_gt[bsIdx][kIdx].cpu().data.numpy().tolist()
        k_error = proc.coord_distance(k_coord, k_gt)
        k_accFlag = eval.acc_pck_pseudo(k_error, [item["norm"] for item in norms if item["imageID"] == imageID][0], args.pck_thr)

        k_augCoords = augPredsArray_mds1[bsIdx][kIdx].cpu().data.numpy().tolist()  # bs,k,df,2
        k_augScores = [self._scoreFormat(self, item) for item in augScoresArray_mds1[0][kIdx].cpu().data.numpy().tolist()]
        k_scores = k_augScores + [self._scoreFormat(self, scores_mds1[0][kIdx].item())]  # bs,k,df
        k_score = k_scores[-1]  # self._calScoreFlag(self, k_scores) -- 各score连乘后的结果。

        k_augCoord = [sum([item[0] for item in k_augCoords]) / len(k_augCoords), sum([item[1] for item in k_augCoords]) / len(k_augCoords)]
        k_intDist = proc.coord_avgDistance(k_augCoords)

        k_sample = {"kpID": k_id, "coord": k_coord, "coord_gt": k_gt, "error": k_error, "acc_flag": k_accFlag, "coords_aug": k_augCoords, "coord_aug": k_augCoord,
                    "scores": k_scores, "score": k_score, "intDist": k_intDist}
        return k_sample

    def _calKSampleExterData(self, mds1_kSample, mds2_kSample, args):
        # cal extDist & aExtDist
        mds1_kSample["extDist"] = mds2_kSample["extDist"] = proc.coord_distance(mds1_kSample["coord"], mds2_kSample["coord"])
        mds1_kSample["aExtDist"] = mds2_kSample["aExtDist"] = proc.coord_distance(mds1_kSample["coord_aug"], mds2_kSample["coord_aug"])
        # cal intDist_lma & extDist_lma & aExtDist_lma
        mds1_kSample = self._lma_calKSampleLMAData(self, mds1_kSample, args.mds1_lma_cache)
        mds2_kSample = self._lma_calKSampleLMAData(self, mds2_kSample, args.mds2_lma_cache)

        mds1_kSample["mixDist"] = mds1_kSample["intDist_lma"] + ((mds1_kSample["extDist_lma"] + mds1_kSample["aExtDist_lma"])/2 if mds1_kSample["extDist_lma"] > 0 else mds1_kSample["aExtDist_lma"])
        mds2_kSample["mixDist"] = mds2_kSample["intDist_lma"] + ((mds2_kSample["extDist_lma"] + mds2_kSample["aExtDist_lma"])/2 if mds2_kSample["extDist_lma"] > 0 else mds2_kSample["aExtDist_lma"])

        mds1_kSample["intDistOK"] = 1 if mds1_kSample["intDist"] <= args.distThrMax else 0
        mds1_kSample["intDistOK_lma"] = 1 if mds1_kSample["intDist_lma"] <= args.distThrMax else 0
        mds1_kSample["extDistOK"] = 1 if mds1_kSample["extDist"] <= args.distThrMax else 0
        mds1_kSample["extDistOK_lma"] = 1 if mds1_kSample["extDist_lma"] <= args.distThrMax else 0
        mds1_kSample["aExtDistOK"] = 1 if mds1_kSample["aExtDist"] <= args.distThrMax else 0
        mds1_kSample["aExtDistOK_lma"] = 1 if mds1_kSample["aExtDist_lma"] <= args.distThrMax else 0
        mds2_kSample["intDistOK"] = 1 if mds2_kSample["intDist"] <= args.distThrMax else 0
        mds2_kSample["intDistOK_lma"] = 1 if mds2_kSample["intDist_lma"] <= args.distThrMax else 0
        mds2_kSample["extDistOK"] = 1 if mds2_kSample["extDist"] <= args.distThrMax else 0
        mds2_kSample["extDistOK_lma"] = 1 if mds2_kSample["extDist_lma"] <= args.distThrMax else 0
        mds2_kSample["aExtDistOK"] = 1 if mds2_kSample["aExtDist"] <= args.distThrMax else 0
        mds2_kSample["aExtDistOK_lma"] = 1 if mds2_kSample["aExtDist_lma"] <= args.distThrMax else 0

        mds1_kSample["unc"] = self._calUncValue(self, mds1_kSample["mixDist"]) if mds1_kSample["intDistOK_lma"] > 0 and mds1_kSample["extDistOK_lma"] > 0 and mds1_kSample["aExtDistOK_lma"] > 0 else 999.0
        mds2_kSample["unc"] = self._calUncValue(self, mds2_kSample["mixDist"]) if mds2_kSample["intDistOK_lma"] > 0 and mds2_kSample["extDistOK_lma"] > 0 and mds2_kSample["aExtDistOK_lma"] > 0 else 999.0
        return mds1_kSample, mds2_kSample

    def getLMAfromCache(self, lma_cache, kpID):
        targets = [item for item in lma_cache if item["kpID"] == kpID]
        if len(targets) == 0:
            item = {"kpID": kpID, "intDist": [], "extDist": [], "aExtDist": [], "intDist_lma": [], "extDist_lma": [], "aExtDist_lma": []}
            lma_cache.append(item)
            return item
        else:
            return targets[0]

    def _calScoreThr(self, dataArray, args):
        scores = [item["score"] for item in dataArray]
        # 方案一：
        scores_sorted = sorted(scores, reverse=True)
        scoreThr = scores_sorted[int((len(scores_sorted) - 1) * 0.5)]
        # 方案二：
        # scoreThr = max(args.scoreThr_min, sum(scores)/len(scores))
        return scoreThr

    def _scoreFormat(self, score):
        return max(0.0, min(1.0, score))

    def _calScoreFlag(self, scores):
        scoreFlag = 1.0
        for score in scores:
            scoreFlag *= score
        return scoreFlag

    def _calUncValue(self, mixDist):
        return 1.0 - math.exp(-mixDist/5)

    def _lma_calKSampleLMAData(self, mds1_kSample, lma_cache):
        lma_target = self.getLMAfromCache(self, lma_cache, mds1_kSample["kpID"])
        lma_target["intDist"].append(mds1_kSample["intDist"])
        lma_target["extDist"].append(mds1_kSample["extDist"])
        lma_target["aExtDist"].append(mds1_kSample["aExtDist"])

        intDist_lma = self._lma_variables(self, lma_target["intDist"])
        extDist_lma = self._lma_variables(self, lma_target["extDist"])
        aExtDist_lma = self._lma_variables(self, lma_target["aExtDist"])

        mds1_kSample["intDist_lma"] = intDist_lma
        mds1_kSample["extDist_lma"] = extDist_lma
        mds1_kSample["aExtDist_lma"] = aExtDist_lma

        lma_target["intDist_lma"].append(intDist_lma)
        lma_target["extDist_lma"].append(extDist_lma)
        lma_target["aExtDist_lma"].append(aExtDist_lma)
        return mds1_kSample

    def _lma_variables(self, sources):
        alphas = [0.5, 0.3, 0.2]
        if len(sources) == 0:
            return 999.0
        elif len(sources) == 1:
            return sources[-1]
        elif len(sources) == 2:
            return sources[-1] * (alphas[0] + alphas[1]) + sources[-2] * alphas[2]
        else:
            return sources[-1] * alphas[0] + sources[-2] * alphas[1] + sources[-3] * alphas[2]