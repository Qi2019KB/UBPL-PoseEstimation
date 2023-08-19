# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
from .process import ProcessUtils as proc


class EvaluationUtils:
    def __init__(self):
        pass

    @classmethod
    def modelSimilarity_byCosineSimilarity(cls, model1, model2):
        v1, v2 = None, None
        for (p1, p2) in zip(model1.parameters(), model2.parameters()):
            if v1 is None and v2 is None:
                v1, v2 = p1.view(-1), p2.view(-1)
            else:
                v1, v2 = torch.cat((v1, p2.view(-1)), 0), torch.cat((v2, p2.view(-1)), 0)  # 拼接
        simVal = 1.0 + (torch.matmul(v1, v2) / (torch.norm(v1) * torch.norm(v2))).item()
        return simVal

    @classmethod
    def modelSimilarity_byDistance(cls, model1, model2):
        sumDistCal = torch.nn.MSELoss(reduction='sum')
        dist_sum = 0.
        for (p1, p2) in zip(model1.parameters(), model2.parameters()):
            dist_sum += sumDistCal(p1, p2).item()
        return dist_sum

    @classmethod
    def predsSimilarity_byDistance(cls, preds_array1, preds_array2):
        n, k, simDis = len(preds_array1), len(preds_array1[0]), 0.
        for nIdx in range(n):
            for kIdx in range(k):
                simDis += proc.coord_distance(preds_array1[nIdx][kIdx], preds_array2[nIdx][kIdx]).item()
        return simDis / (n * k)

    @classmethod
    def uncertainty_fromDistance(cls, preds_mul, preds_mean):
        mc, bs, k, _ = preds_mul.shape
        bsDists = []
        for iIdx in range(bs):
            kDists = []
            for kIdx in range(k):
                dists = []
                for mcIdx in range(mc):
                    p1 = preds_mul[mcIdx][iIdx][kIdx].cpu().numpy().tolist()
                    p2 = preds_mean[iIdx][kIdx].cpu().numpy().tolist()
                    dist = cls._calDist_fromCoords(cls, p1, p2)
                    dists.append(dist)
                dist_avg = sum(dists)/len(dists)
                kDists.append(torch.tensor(dist_avg))
            bsDists.append(torch.stack(kDists, 0))
        unc = torch.stack(bsDists, 0)
        unc = unc/unc.max()
        uncW = (-unc).exp()
        return unc, uncW

    @classmethod
    def err_kps(self, preds, gts):
        bs, k, _ = preds.shape
        kpsErrorArray = []
        for i in range(bs):
            kpsError = []
            for j in range(k):
                kpsError.append(proc.coord_distance(preds[i][j][0:2], gts[i][j][0:2]))
            kpsErrorArray.append(torch.stack(kpsError, 0))
        samplesError = torch.stack(kpsErrorArray, 0)
        return samplesError

    @classmethod
    def error_kps_mul(cls, preds_mul, gts):
        mc, bs, k, _ = preds_mul.shape
        return torch.stack([cls.err_kps(preds_mul[mcIdx], gts) for mcIdx in range(mc)], 0)

    @classmethod
    def acc_pck_pseudo_norm(cls, imageIDs, gts, pck_ref):
        bsNum, gts_list = gts.size(0), gts.cpu().data.numpy().tolist()
        norms = []
        for bsIdx in range(bsNum):
            coord1, coord2 = gts_list[bsIdx][pck_ref[0]], gts_list[bsIdx][pck_ref[1]]
            norms.append({"imageID": imageIDs[bsIdx],
                          "norm": proc.coord_distance(coord1, coord2)})
        return norms

    @classmethod
    def acc_pck_pseudo(cls, error, norm, pck_thr):
        return 1 if error/norm < pck_thr else 0

    @classmethod
    def acc_pck(cls, preds, gts, pck_ref, pck_thr):
        bs, k, _ = preds.shape
        # 计算各点的相对距离
        dists, dists_ref = cls._acc_calDists(cls, preds, gts, pck_ref)

        # 计算error
        errs, err_sum, err_num = torch.zeros(k + 1), 0, 0
        for kIdx in range(k):
            if errs[kIdx] >= 0:  # 忽略带有-1的值
                errs[kIdx] = dists[kIdx].sum() / len(dists[kIdx])
                err_sum += errs[kIdx]
                err_num += 1
        errs[-1] = err_sum / err_num

        # 根据thr计算accuracy
        accs, acc_sum, acc_num = torch.zeros(k + 1), 0, 0
        for kIdx in range(k):
            accs[kIdx] = cls._acc_counting(cls, dists_ref[kIdx], pck_thr)
            if accs[kIdx] >= 0:  # 忽略带有-1的值
                acc_sum += accs[kIdx]
                acc_num += 1
        if acc_num != 0:
            accs[-1] = acc_sum / acc_num
        return errs, accs

    # 计算各点的相对距离
    def _acc_calDists(self, preds, gts, pckRef_idxs):
        # 计算参考距离（基于数据集的参考关键点对）
        bs, k, _ = preds.shape
        dists, dists_ref = torch.zeros(k, bs), torch.zeros(k, bs)
        for iIdx in range(bs):
            norm = torch.dist(gts[iIdx, pckRef_idxs[0], 0:2], gts[iIdx, pckRef_idxs[1], 0:2])
            for kIdx in range(k):
                if gts[iIdx, kIdx, 0] > 1 and gts[iIdx, kIdx, 1] > 1:
                    dists[kIdx, iIdx] = torch.dist(preds[iIdx, kIdx, 0:2], gts[iIdx, kIdx, 0:2])
                    dists_ref[kIdx, iIdx] = torch.dist(preds[iIdx, kIdx, 0:2], gts[iIdx, kIdx, 0:2]) / norm
                else:
                    dists[kIdx, iIdx] = -1
                    dists_ref[kIdx, iIdx] = -1
        return dists, dists_ref

    # 返回低于阈值的百分比
    def _acc_counting(cls, dists, thr=0.5):
        dists_plus = dists[dists != -1]
        if len(dists_plus) > 0:
            return 1.0 * (dists_plus < thr).sum().item() / len(dists_plus)
        else:
            return -1

    # 计算欧式距离--坐标
    def _calDist_fromCoords(self, coord1, coord2):
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** (1 / 2)

    # 计算欧氏距离--多维向量
    def _euclidean_dist(self, v1, v2):
        # x: N x D
        # y: M x D
        n = v1.size(0)
        m = v2.size(0)
        d = v1.size(1)
        assert d == v2.size(1)
        v1 = v1.unsqueeze(1).expand(n, m, d)
        v2 = v2.unsqueeze(0).expand(n, m, d)
        return torch.pow(v1 - v2, 2).sum(2)

    # region function：方差计算，有问题（废弃）
    # params：
    #   （1）multiPreds：多重预测结果集合
    # return：
    #   （1）uncVal：不确定性值
    # endregion
    @classmethod
    def uncertainty_fromFeatures_discard(cls, outs_mul):
        uncVal, uncMean, uncVar = None, None, None
        mc, bs, k, _, _ = outs_mul.shape
        if mc > 1:
            predsArray = [F.softmax(preds.detach().view(bs, k, -1), dim=2) for preds in outs_mul]
            sampleMultiUncArray = []
            for iIdx in range(bs):
                kpsMultiUncArray = []
                for kIdx in range(k):
                    kpMultiUncArray = []
                    for v1Idx in range(mc):
                        if v1Idx+1 == mc: break
                        for v2Idx in range(v1Idx+1, mc):
                            v1, v2 = predsArray[v1Idx][iIdx][kIdx], predsArray[v2Idx][iIdx][kIdx]
                            kpUnc = 1 - torch.matmul(v1, v2) / (torch.norm(v1) * torch.norm(v2))
                            kpMultiUncArray.append(kpUnc)
                    kpsMultiUncArray.append(torch.stack(kpMultiUncArray, 0))
                sampleMultiUncArray.append(torch.stack(kpsMultiUncArray, 0))
            uncVal = torch.stack(sampleMultiUncArray, 0)
            uncMean = torch.mean(uncVal, dim=2)
            uncVar = torch.var(uncVal, dim=2)
        return uncVal, uncMean, uncVar
