import torch
from torch import nn
from models.base.layers import Conv, Hourglass, Pool, Residual, Merge


# function: 基于AMIL的Hourglass网络
class StackedHourglass(nn.Module):
    # function: 初始化
    # params:
    #   （1）k：关节点个数
    #   （2）nStack：stacked-hourglass模块的堆叠数量
    def __init__(self, k, nStack, mode):
        super(StackedHourglass, self).__init__()
        # 1. 参数设置
        self.k = k
        self.nStack = nStack
        self.mode = mode

        # 2. Pose网络初始化
        # 2.1 通道、特征尺寸对齐（batchSize*3*256*256 ==> batchSize*256*64*64）
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, 256)
        )

        # 2.2 4-Stacked_Hourglass
        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, 256, False, 0)
            ) for i in range(self.nStack)])

        # 2.3 特征提取
        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(256, 256),
                Conv(256, 256, 1, bn=True, relu=True)
            ) for i in range(self.nStack)])

        # 2.4 Pose预测
        self.preds = nn.ModuleList([Conv(256, self.k, 1, relu=False, bn=False) for i in range(self.nStack)])

        # 2.5 feature融合
        self.merge_features = nn.ModuleList(
            [Merge(256, 256) for i in range(self.nStack - 1)])

        # 2.6 pred融合
        self.merge_preds = nn.ModuleList(
            [Merge(self.k, 256) for i in range(self.nStack - 1)])

        self.projection = self._get_projection(self.mode)

    # function: 前馈
    # params:
    #   （1）imgs：图像数据
    # return:
    #   （1）preds：预测结果
    def forward(self, imgs):
        # 1. 图像数据预处理（对齐通道），输出尺寸：batchSize*256*64*64
        x = self.pre(imgs)

        # 2. nStack次shg处理
        combined_hm_preds, combined_features = [], []
        for i in range(self.nStack):
            # 2.1 通过4-stacked_hourglass模块，输出尺寸：batchSize*256*64*64
            hg = self.hgs[i](x)

            # 2.2 进行特征提取模块，输出尺寸：batchSize*256*64*64
            feature = self.features[i](hg)

            if self.mode != "default": combined_features.append(self.projection(feature))

            # 2.3 进行pose预测，输出尺寸：batchSize*k*64*64
            preds = self.preds[i](feature)

            # 2.4 存储pose预测结果
            combined_hm_preds.append(preds)

            # 2.5 特征融合（x、feature、pred），输出尺寸：batchSize*256*64*64
            if i < self.nStack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        if self.mode == "default":
            preds = torch.stack(combined_hm_preds, 1)
            return preds
        else:
            preds, features = torch.stack(combined_hm_preds, 1), torch.stack(combined_features, 1)
            return preds, features

    def _get_projection(self, mode):
        if mode == "MaxPool":
            return nn.Sequential(nn.MaxPool2d(2, 2))
        elif mode == "AvgPool":
            return nn.Sequential(nn.AvgPool2d(2, 2))
        elif self.mode == "ConvOne":
            return nn.Sequential(nn.Conv2d(128, 128, 1, 1, 0, bias=False))


def hg(k, nStack=3, mode="default", nograd=False):
    model = StackedHourglass(k, nStack, mode).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model
