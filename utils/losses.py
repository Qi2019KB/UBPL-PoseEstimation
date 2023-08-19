# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from .process import ProcessUtils as proc


class JointMSELoss(nn.Module):
    def __init__(self, nStack=1, useKPsGate=False, useSampleWeight=False):
        super(JointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.useKPsGate = useKPsGate
        self.useSampleWeight = useSampleWeight

    def forward(self, preds, gts, kpsGate=None, sampleWeight=None):
        bs, k = preds.size(0), preds.size(1) if self.nStack == 1 else preds.size(2)
        kpsGate_clone = torch.ones([bs, k]) if kpsGate is None else kpsGate.detach()
        kpsNum = proc.kps_getLabeledCount(kpsGate_clone)
        combined_loss = []
        for nIdx in range(self.nStack):
            v1 = preds.reshape((bs, k, -1)) if self.nStack == 1 else preds[:, nIdx].reshape((bs, k, -1))
            v2 = gts.reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if self.useKPsGate: loss = loss.mul(kpsGate_clone)
            if self.useSampleWeight and sampleWeight is not None: loss = loss.mul(sampleWeight)
            combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss.sum(), self.nStack*kpsNum


class JointDistLoss(nn.Module):
    def __init__(self, nStack=1, useKPsGate=False, useSampleWeight=False):
        super(JointDistLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.useKPsGate = useKPsGate
        self.useSampleWeight = useSampleWeight

    def forward(self, preds1, preds2, kpsGate=None, sampleWeight=None):
        bs, k = preds1.size(0), preds1.size(1) if self.nStack == 1 else preds1.size(2)
        kpsGate_clone = torch.ones([bs, k]) if kpsGate is None else kpsGate.detach()
        kpsNum = proc.kps_getLabeledCount(kpsGate_clone)
        combined_loss = []
        for nIdx in range(self.nStack):
            v1 = preds1.reshape((bs, k, -1)) if self.nStack == 1 else preds1[:, nIdx].reshape((bs, k, -1))
            v2 = preds2.reshape((bs, k, -1)) if self.nStack == 1 else preds2[:, nIdx].reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if self.useKPsGate: loss = loss.mul(kpsGate_clone)
            if self.useSampleWeight and sampleWeight is not None: loss = loss.mul(sampleWeight)
            combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss.sum(), self.nStack*kpsNum


class JointFeatureDistLoss(nn.Module):
    def __init__(self):
        super(JointFeatureDistLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, inp1, inp2):
        bs, n, c, h, w = inp1.size()
        combined_loss = []
        for nIdx in range(n):
            v1 = inp1[:, nIdx].reshape((bs, c, -1))
            v2 = inp2[:, nIdx].reshape((bs, c, -1))
            loss = self.criterion(v1, v2).mean(-1)
            combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss.sum(), bs*n


class JointPseudoLoss(nn.Module):
    def __init__(self, nStack=1, scoreThr=0.8):
        super(JointPseudoLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.scoreThr = scoreThr

    def forward(self, preds, targets, sampleWeight):
        bs, k = preds.size(0), preds.size(1) if self.nStack == 1 else preds.size(2)
        combined_loss = []
        targets_mean = torch.mean((targets if self.nStack == 1 else targets[:, :, -1]), dim=0)
        num_pseudo, num_selected, joint_score_mean = 0, 0, []
        for nIdx in range(self.nStack):
            v1 = preds.reshape((bs, k, -1)) if self.nStack == 1 else preds[:, nIdx].reshape((bs, k, -1))
            v2 = targets_mean.reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if sampleWeight is not None: loss = loss.mul(sampleWeight)
            # region cal keypoint weight from confidence score 在所有样本中，选heatmap中最大值大的。
            v1_softmax = torch.softmax(v1, dim=-2)
            v1_score, _ = torch.max(v1_softmax, dim=-1)
            v1_mask = v1_score.ge(self.scoreThr).float()

            v2_softmax = torch.softmax(v2, dim=-2)
            v2_score, _ = torch.max(v2_softmax, dim=-1)
            v2_mask = v2_score.ge(self.scoreThr).float()

            mask = v1_mask.mul(v2_mask)
            num_pseudo += len([l for l in loss.reshape(-1) if l > 0])
            num_selected += len([item for item in mask.reshape(-1) if item > 0])
            v1_score_unlabeled, v2_score_unlabeled = [], []
            for swIdx, sw in enumerate(sampleWeight.gt(0).float()):
                if sw > 0:
                    v1_score_unlabeled.append(v1_score[swIdx])
                    v2_score_unlabeled.append(v2_score[swIdx])
            v1_score_unlabeled = torch.stack(v1_score_unlabeled, dim=0)
            v2_score_unlabeled = torch.stack(v2_score_unlabeled, dim=0)
            joint_score_mean.append((v1_score_unlabeled.mean(-2) + v2_score_unlabeled.mean(-2))/2)
            combined_loss.append(loss*mask)
            # endregion
            # combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        joint_score_mean = torch.stack(joint_score_mean, dim=0).mean(0)
        return combined_loss.sum(), num_pseudo, num_selected, joint_score_mean


class JointPseudoLoss2(nn.Module):
    def __init__(self, nStack=1, selRate=0.5):
        super(JointPseudoLoss2, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.selRate = selRate

    def forward(self, preds, targets, sampleWeight):
        bs, k = preds.size(0), preds.size(1) if self.nStack == 1 else preds.size(2)
        combined_loss = []
        targets_mean = torch.mean((targets if self.nStack == 1 else targets[:, :, -1]), dim=0)
        num_pseudo, num_selected, joint_score_mean = 0, 0, []
        v1_rateThrs, v2_rateThrs = [], []
        for nIdx in range(self.nStack):
            v1 = preds.reshape((bs, k, -1)) if self.nStack == 1 else preds[:, nIdx].reshape((bs, k, -1))
            v2 = targets_mean.reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if sampleWeight is not None: loss = loss.mul(sampleWeight)
            # region cal keypoint weight from confidence score 在所有样本中，选heatmap中最大值大的。
            v1_softmax = torch.softmax(v1, dim=-2)
            v1_score, _ = torch.max(v1_softmax, dim=-1)
            v1_rateThr = torch.sort(v1_score.reshape(-1), dim=0)[0][int(len(v1_score.reshape(-1))*(1-self.selRate))]
            v1_rateThrs.append(v1_rateThr)
            v1_mask = v1_score.ge(v1_rateThr).float()

            v2_softmax = torch.softmax(v2, dim=-2)
            v2_score, _ = torch.max(v2_softmax, dim=-1)
            v2_rateThr = torch.sort(v2_score.reshape(-1), dim=0)[0][int(len(v2_score.reshape(-1))*(1-self.selRate))]
            v2_rateThrs.append(v2_rateThr)
            v2_mask = v2_score.ge(v2_rateThr).float()

            mask = v1_mask.mul(v2_mask)
            num_pseudo += len([l for l in loss.reshape(-1) if l > 0])
            num_selected += len([item for item in mask.reshape(-1) if item > 0])
            v1_score_unlabeled, v2_score_unlabeled = [], []
            for swIdx, sw in enumerate(sampleWeight.gt(0).float()):
                if sw > 0:
                    v1_score_unlabeled.append(v1_score[swIdx])
                    v2_score_unlabeled.append(v2_score[swIdx])
            v1_score_unlabeled = torch.stack(v1_score_unlabeled, dim=0)
            v2_score_unlabeled = torch.stack(v2_score_unlabeled, dim=0)
            joint_score_mean.append((v1_score_unlabeled.mean(-2) + v2_score_unlabeled.mean(-2))/2)
            combined_loss.append(loss*mask)
            # endregion
            # combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        joint_score_mean = torch.stack(joint_score_mean, dim=0).mean(0)
        rateThr1, rateThr2 = torch.stack(v1_rateThrs, dim=0), torch.stack(v2_rateThrs, dim=0)
        return combined_loss.sum(), num_pseudo, num_selected, joint_score_mean, rateThr1, rateThr2


class JointPseudoLoss3(nn.Module):
    def __init__(self, nStack=1, scoreThr=0.5):
        super(JointPseudoLoss3, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.scoreThr = scoreThr

    def forward(self, preds, targets, sampleWeight):
        bs, k = preds.size(0), preds.size(1) if self.nStack == 1 else preds.size(2)
        combined_loss = []
        targets_mean = torch.mean((targets if self.nStack == 1 else targets[:, :, -1]), dim=0)
        num_pseudo, num_selected, joint_score_mean = 0, 0, []
        for nIdx in range(self.nStack):
            v1 = preds.reshape((bs, k, -1)) if self.nStack == 1 else preds[:, nIdx].reshape((bs, k, -1))
            v2 = targets_mean.reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if sampleWeight is not None: loss = loss.mul(sampleWeight)
            # region cal keypoint weight from confidence score 在所有样本中，选heatmap中最大值大的。
            v1_score, _ = torch.max(v1, dim=-1)
            v1_mask = v1_score.ge(self.scoreThr).float()

            v2_score, _ = torch.max(v2, dim=-1)
            v2_mask = v2_score.ge(self.scoreThr).float()

            mask = v1_mask.mul(v2_mask)
            num_pseudo += len([l for l in loss.reshape(-1) if l > 0])
            num_selected += len([item for item in mask.reshape(-1) if item > 0])
            v1_score_unlabeled, v2_score_unlabeled = [], []
            for swIdx, sw in enumerate(sampleWeight.gt(0).float()):
                if sw > 0:
                    v1_score_unlabeled.append(v1_score[swIdx])
                    v2_score_unlabeled.append(v2_score[swIdx])
            v1_score_unlabeled = torch.stack(v1_score_unlabeled, dim=0)
            v2_score_unlabeled = torch.stack(v2_score_unlabeled, dim=0)
            joint_score_mean.append((v1_score_unlabeled.mean(-2) + v2_score_unlabeled.mean(-2))/2)
            combined_loss.append(loss*mask)
            # endregion
            # combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        joint_score_mean = torch.stack(joint_score_mean, dim=0).mean(0)
        rateThr1, rateThr2 = self.scoreThr, self.scoreThr
        return combined_loss.sum(), num_pseudo, num_selected, joint_score_mean, rateThr1, rateThr2


class JointDistLoss_mt(nn.Module):
    def __init__(self, nStack=1, useKPsGate=False, useSampleWeight=False, selRate=0.5):
        super(JointDistLoss_mt, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.useKPsGate = useKPsGate
        self.useSampleWeight = useSampleWeight
        self.selRate = selRate

    def forward(self, preds1, preds2, kpsGate=None, sampleWeight=None):
        bs, k = preds1.size(0), preds1.size(1) if self.nStack == 1 else preds1.size(2)
        kpsGate_clone = torch.ones([bs, k]) if kpsGate is None else kpsGate.detach()
        kpsNum = proc.kps_getLabeledCount(kpsGate_clone)
        combined_loss = []
        for nIdx in range(self.nStack):
            v1 = preds1.reshape((bs, k, -1)) if self.nStack == 1 else preds1[:, nIdx].reshape((bs, k, -1))
            v2 = preds2.reshape((bs, k, -1)) if self.nStack == 1 else preds2[:, nIdx].reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if self.useKPsGate:
                loss = loss.mul(kpsGate_clone)
            if self.useSampleWeight and sampleWeight is not None:
                loss = loss.mul(sampleWeight)
            # region cal keypoint weight from confidence score 在所有样本中，选heatmap中最大值大的。
            v2_softmax = torch.softmax(v2, dim=-2)
            v2_score, _ = torch.max(v2_softmax, dim=-1)
            v2_rateThr = torch.sort(v2_score.reshape(-1), dim=0)[0][int(len(v2_score.reshape(-1))*(1-self.selRate))]
            v2_mask = v2_score.ge(v2_rateThr).float()
            # endregion
            combined_loss.append(loss*v2_mask)
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss.sum(), self.nStack * kpsNum


class JointDistLoss_mt2(nn.Module):
    def __init__(self, nStack=1, useKPsGate=False, useSampleWeight=False, scoreThr=0.5):
        super(JointDistLoss_mt2, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.useKPsGate = useKPsGate
        self.useSampleWeight = useSampleWeight
        self.scoreThr = scoreThr

    def forward(self, preds1, preds2, kpsGate=None, sampleWeight=None):
        bs, k = preds1.size(0), preds1.size(1) if self.nStack == 1 else preds1.size(2)
        kpsGate_clone = torch.ones([bs, k]) if kpsGate is None else kpsGate.detach()
        kpsNum = proc.kps_getLabeledCount(kpsGate_clone)
        combined_loss = []
        num_pseudo, num_selected, joint_score_mean = 0, 0, []
        for nIdx in range(self.nStack):
            v1 = preds1.reshape((bs, k, -1)) if self.nStack == 1 else preds1[:, nIdx].reshape((bs, k, -1))
            v2 = preds2.reshape((bs, k, -1)) if self.nStack == 1 else preds2[:, nIdx].reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if self.useKPsGate:
                loss = loss.mul(kpsGate_clone)
            if self.useSampleWeight and sampleWeight is not None:
                loss = loss.mul(sampleWeight)
            # region cal keypoint weight from confidence score 在所有样本中，选heatmap中最大值大的。
            v2_score, _ = torch.max(v2, dim=-1)
            v2_mask = v2_score.ge(self.scoreThr).float()

            num_pseudo += len([l for l in loss.reshape(-1) if l > 0])
            num_selected += len([item for item in v2_mask.reshape(-1) if item > 0])
            v2_score_unlabeled = []
            for swIdx, sw in enumerate(sampleWeight.gt(0).float()):
                if sw > 0:
                    v2_score_unlabeled.append(v2_score[swIdx])
            v2_score_unlabeled = torch.stack(v2_score_unlabeled, dim=0)
            joint_score_mean.append(v2_score_unlabeled.mean(-2))

            # endregion
            combined_loss.append(loss*v2_mask)
        combined_loss = torch.stack(combined_loss, dim=1)
        joint_score_mean = torch.stack(joint_score_mean, dim=0).mean(0)
        return combined_loss.sum(), self.nStack * kpsNum, num_pseudo, num_selected, joint_score_mean


class ClassLoss(nn.Module):
    def __init__(self, useSampleWeight=False):
        super(ClassLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.useSampleWeight = useSampleWeight

    def forward(self, preds, labels, sampleWeight=None):
        loss = self.criterion(preds, labels)
        if self.useSampleWeight and sampleWeight is not None:
            loss = loss.mul(sampleWeight.squeeze())
            return loss.sum(), len([label for label in labels if label >= 0])
        else:
            return loss.sum(), len([label for label in labels if label >= 0])


class ClassDistLoss(nn.Module):
    def __init__(self):
        super(ClassDistLoss, self).__init__()

    def forward(self, pred1, pred2):
        assert pred1.size() == pred2.size()
        bs, _ = pred1.shape
        pred1_softmax = F.softmax(pred1, dim=1)
        pred2_softmax = F.softmax(pred2, dim=1)
        num_classes = pred1.size()[1]
        return F.mse_loss(pred1_softmax, pred2_softmax, size_average=False) / num_classes, bs


class ClassSymDistLoss(nn.Module):
    def __init__(self):
        super(ClassSymDistLoss, self).__init__()

    def forward(self, pred1, pred2):
        assert pred1.size() == pred2.size()
        bs, _ = pred1.shape
        num_classes = pred1.size()[1]
        return torch.sum((pred1 - pred2) ** 2) / num_classes, bs


class ClassPseudoLoss(nn.Module):
    def __init__(self):
        super(ClassPseudoLoss, self).__init__()

    def forward(self, preds, targets, sampleWeight):
        targets_softmax = torch.mean(torch.stack([F.softmax(target, dim=1) for target in targets], dim=0), dim=0)
        assert preds.size() == targets_softmax.size()
        bs, _ = preds.shape
        preds_softmax = F.softmax(preds, dim=1)

        criterion = nn.MSELoss(reduction='none')
        loss = torch.mean(criterion(preds_softmax, targets_softmax), dim=-1).mul(sampleWeight.squeeze())
        n = len([w for w in sampleWeight if w > 0])
        return loss.sum(), n


class ClassFeatureDistLoss(nn.Module):
    def __init__(self):
        super(ClassFeatureDistLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, preds1, preds2):
        bs, c, _, _ = preds1.shape
        v1, v2 = preds1.clone().reshape((bs, c, -1)), preds2.clone().reshape((bs, c, -1))
        dists = self.criterion(v1, v2).mean(-1).mean(-1)
        c_cov = 1/dists
        return c_cov.sum(), bs


class AvgCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 0. if self.count == 0 else self.sum / self.count


class AvgCounters(object):
    def __init__(self, num=1):
        self.counters = [AvgCounter() for i in range(num)]
        self.reset()

    def reset(self):
        for counter in self.counters:
            counter.reset()

    def update(self, idx, val, n=1):
        self.check_idx(idx)
        self.counters[idx].update(val, n)

    def avg(self):
        return [item.avg for item in self.counters]

    def sum(self):
        return [item.sum for item in self.counters]

    def check_idx(self, idx):
        if len(self.counters) < idx + 1:
            for i in range(len(self.counters), idx + 1):
                self.counters.append(AvgCounter())


