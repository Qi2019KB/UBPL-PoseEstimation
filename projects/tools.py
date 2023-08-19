# -*- coding: utf-8 -*-
import math
import copy
import torch
from utils.process import ProcessUtils as proc
from utils.evaluation import EvaluationUtils as eval


class ProjectTools:
    def __init__(self):
        pass

    @classmethod
    def getSampleWeight(cls, isLabeledArray, args):
        isLabeledArray = [islabeled.to(args.device, non_blocking=True) for islabeled in isLabeledArray]  # 2 * [bs]
        weights = [islabeled.detach().float() for islabeled in isLabeledArray]  # 2 * [bs]
        weights_pseudo = [0. * torch.ones_like(sampleWeight) for sampleWeight in weights]  # 2 * [bs]
        weights = [cls.setVariable(torch.where(isLabeledArray[idx] > 0, weights[idx], weights_pseudo[idx]), args.device).unsqueeze(-1) for idx in range(len(isLabeledArray))]
        return weights

    @classmethod
    def getSampleWeight_nega(cls, isLabeledArray, args):
        isLabeledArray = [islabeled.to(args.device, non_blocking=True) for islabeled in isLabeledArray]  # 2 * [bs]
        weights = [islabeled.detach().float() for islabeled in isLabeledArray]  # 2 * [bs]
        weights_pseudo = [args.pseudoWeight * torch.ones_like(sampleWeight) for sampleWeight in weights]  # 2 * [bs]
        weights_zero = [0. * torch.ones_like(sampleWeight) for sampleWeight in weights]
        weights = [cls.setVariable(torch.where(isLabeledArray[idx] > 0, weights_zero[idx], weights_pseudo[idx]), args.device).unsqueeze(-1) for idx in range(len(isLabeledArray))]
        return weights

    @classmethod
    def getSampleWeight_mt(cls, islabeled, args):
        islabeled = islabeled.to(args.device, non_blocking=True)  # [bs]
        weight = islabeled.detach().float()  # [bs]
        weight_pseudo = 0. * torch.ones_like(weight)  # [bs]
        weight = cls.setVariable(torch.where(islabeled > 0, weight, weight_pseudo), args.device).unsqueeze(-1)
        return weight

    @classmethod
    def getSampleWeight_mt_nega(cls, islabeled, args):
        islabeled = islabeled.to(args.device, non_blocking=True)  # [bs]
        weight = islabeled.detach().float()  # [bs]
        weight_pseudo = args.pseudoWeight * torch.ones_like(weight)  # [bs]
        weight_zero = 0. * torch.ones_like(weight)  # [bs]
        weight = cls.setVariable(torch.where(islabeled > 0, weight_zero, weight_pseudo), args.device).unsqueeze(-1)
        return weight

    @classmethod
    def getSampleWeight_mt_cons(cls, islabeled, args):
        islabeled = islabeled.to(args.device, non_blocking=True)  # [bs]
        weight = islabeled.detach().float()  # [bs]
        weight_pseudo = args.pseudoWeight * torch.ones_like(weight)  # [bs]
        weight_zero = 1. * torch.ones_like(weight)  # [bs]
        weight = cls.setVariable(torch.where(islabeled > 0, weight_zero, weight_pseudo), args.device).unsqueeze(-1)
        return weight

    @classmethod
    def setVariable(cls, tensor, deviceID, toVariable=True, requires_grad=True):
        if toVariable:
            return torch.autograd.Variable(tensor.to(deviceID, non_blocking=True), requires_grad=requires_grad)
        else:
            return tensor.to(deviceID, non_blocking=True)

    @classmethod
    def setContent(cls, dataArray, fmt):
        strContent = ""
        for dataIdx, dataItem in enumerate(dataArray):
            if dataIdx == len(dataArray)-1:
                strContent += "{}".format(format(dataItem, fmt))
            else:
                strContent += "{}, ".format(format(dataItem, fmt))
        return strContent
