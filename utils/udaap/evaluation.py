from __future__ import absolute_import

import math
import numpy as np
import torch

from .transforms import transform_preds

__all__ = ['accuracy', 'AverageMeter']


# 从预测结果中获取点坐标值，通过confidence score量化的一个是否使用的mask来评价是否采用该关键点的预测结果（小于0的则认为不可信，不采用）
def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'  # torch.Size([1, 18, 64, 64])
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)  # scores_view's torch.Size([1, 18, 4096])，在index=2的维度取最大值及其索引。

    maxval = maxval.view(scores.size(0), scores.size(1), 1)  # maxval's torch.Size([1, 18])，在index=2处添加一个维度，形成：torch.Size([1, 18, 1])
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1  # idx's torch.Size([1, 18])，在index=2处添加一个维度，形成：torch.Size([1, 18, 1])。此外，index从0开始，所以所有索引加1。
    # tensor.repeat()：https://blog.csdn.net/qq_29695701/article/details/89763168
    preds = idx.repeat(1, 1, 2).float()  # 在index=2的维度上增加一个相同的值。由idx's torch.Size([1, 18, 1]) ==> preds's torch.Size([1, 18, 1])
    # 将各点预测最高点的索引集合从1*4096的特征图映射到64*64的特征图中
    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1  # 在64*64的特征图中，计算其列数（y）
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1  # 在64*64的特征图中，计算其行数（x）
    # tensor.gt(0)：各项与0比，大于0则为True，小于则为False
    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()  # 与preds对应，形成一个由confidence score量化的一个是否使用的mask（小于0的则认为不可信，不采用）
    preds *= pred_mask
    return preds


def get_preds_all(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1
    # pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    # preds *= pred_mask
    return preds


def mpii_calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists


def mpii_dist_acc(dist, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1


def mpii_accuracy(output, target, idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]-1])
        if acc[i+1] >= 0:
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc


# 计算预测坐标和真值坐标之间的距离（除以normalize项）
def calc_dists(preds, target, normalize):
    preds = preds.float()  # 预测坐标
    target = target.float()  # 真值坐标
    dists = torch.zeros(preds.size(1), preds.size(0))  # torch.Size([18, 1])
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1 and normalize[n] > 0:  # 坐标合法化检查
                dists[c, n] = torch.dist(preds[n, c, :], target[n, c, :])/normalize[n]  # 点坐标之间的距离，除以标准化项。
            else:
                dists[c, n] = -1
    return dists

# 返回低于阈值的百分比（批次中，以关键点为单位，统计各image中各点误差值低于thr的数量/总数量，肯定是越高越好），忽略带有-1的值
def dist_acc(dist, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1


def calc_metrics(dists, idxs, path='', category=''):

    idxs_array = np.array(idxs)-1
    dists_pickup = dists[idxs_array, :]
    errors = dists_pickup[dists_pickup!=-1]

    axes1 = np.linspace(0, 1, 100)
    axes2 = np.zeros(100)

    for i in range(100):
        axes2[i] = (errors < axes1[i]).sum().float() / float(errors.size(0))

    auc = round(np.sum(axes2[1:81]) / .8, 2)
    return auc

# 根据PCK计算精度，但使用地面真相热图而不是x,y位置。第一个返回的值是“idxs”之间的平均精度，然后是各个精度。
# Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
# First value to be returned is average accuracy across 'idxs', followed by individual accuracies
def accuracy(output, target, idxs, thr=0.5):
    preds   = get_preds(output)  # torch.Size([bat, 18, 2])
    gts     = get_preds(target)  # torch.Size([bat, 18, 2])
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    # 计算预测坐标和真值坐标之间的距离（除以normalize项）
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        # 返回低于阈值的百分比（批次中，以关键点为单位，统计各image中各点误差值低于thr的数量/总数量，肯定是越高越好），忽略带有-1的值
        acc[i+1] = dist_acc(dists[idxs[i]-1])
        if acc[i+1] >= 0:  # 忽略带有-1的值
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        # 第一个返回的值是“idxs”之间的平均精度，然后是各个精度
        acc[0] = avg_acc / cnt  # 将平均值加到array的第一项。
    # acc：pck@0.5；dists：各image中，各点的预测与真值坐标的距离差。
    return acc, dists


def accuracy_2animal(output, target, output_idxs, target_idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    # pick up idxs
    output_idxs_array = np.array(output_idxs)-1
    target_idxs_array = np.array(target_idxs)-1

    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10

    preds = preds[:, output_idxs_array, :]
    gts = gts[:, target_idxs_array, :]

    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(target_idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(target_idxs)):
        acc[i+1] = dist_acc(dists[target_idxs_array[i]])
        if acc[i+1] >= 0:
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc, dists


def calc_metrics_2animal(dists, path='', category=''):
    perpoint_error = torch.zeros(dists.size(0))
    for j in range(dists.size(0)):
        dist = dists[j,:][dists[j,:]!=-1]
        perpoint_error[j] = torch.mean(dist.view(-1),0).view(-1)
    print(perpoint_error)

    errors = dists[dists!=-1]

    axes1 = np.linspace(0, 1, 100)
    axes2 = np.zeros(100)

    for i in range(100):
        axes2[i] = (errors < axes1[i]).sum().float() / float(errors.size(0))

    auc = round(np.sum(axes2[1:81]) / .8, 2)
    return auc


def final_preds(output, center, scale, res):
    coords = get_preds(output) # float type

    # # pose-processing -- 有问题，导致预测结果向右下偏移。
    # for n in range(coords.size(0)):  # coords' torch.Size([1, 18, 2])
    #     for p in range(coords.size(1)):
    #         hm = output[n][p]  # 关键点对应的64*64特征图。
    #         px = int(math.floor(coords[n][p][0]))  # 关键点的位置中的x。
    #         py = int(math.floor(coords[n][p][1]))  # 关键点的位置中的y。
    #         if px > 1 and px < res[0] and py > 1 and py < res[1]:  # 验证预测位置的合法性
    #             # 计算预测位置的四周值的差值。
    #             diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
    #             coords[n][p] += diff.sign() * .25  # 对预测位置的四周值做了一个附加的评估，对应调整下位置。 diff.sign()：大于0则为1，小于0则为-1
    # coords += 0.5  # 没看懂，为什么加。
    preds = coords.clone()

    # 对预测结果进行仿射变换（对关键点的预测坐标值进行处理）（Transform back）
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count
