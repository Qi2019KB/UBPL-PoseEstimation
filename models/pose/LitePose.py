import torch.nn as nn


# it must be used always inside a Sequential()
class ConvBlockBase(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=True):
        if (activation):
            super(ConvBlockBase, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            super(ConvBlockBase, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
            )


# 基本的卷积单元，含残差结构。 -- Wu
class ConvMobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvMobileBlock, self).__init__()

        self.useResidual = in_channels == out_channels and stride == 1

        midChannels = in_channels + out_channels // 2

        self.s = nn.Sequential(
            ConvBlockBase(in_channels, midChannels, 1, activation=False),
            ConvBlockBase(midChannels, midChannels, kernel_size, stride, activation=False),
            ConvBlockBase(midChannels, out_channels, 1)
        )

    def forward(self, x):
        return self.s(x) + x if self.useResidual else self.s(x)

    # 卷积Stage模块。在预定义好的网络配置config中，根据配置，生成指定结构的特征提取网络. -- Wu


class ConvStage(nn.Module):
    def __init__(self, stageConfig):
        super(ConvStage, self).__init__()
        listBlock = []
        for i in range(len(stageConfig)):
            listBlock.append(ConvMobileBlock(stageConfig[i][0], stageConfig[i][1], stageConfig[i][2], stageConfig[i][3]))
        self.stage = nn.Sequential(*listBlock)

    def forward(self, x):
        return self.stage(x)


class LitePose(nn.Module):
    def __init__(self, kpsCount, mode) -> None:
        super().__init__()
        self.kpsCount = kpsCount
        self.mode = mode
        self.nStack = 1
        in_channels = 16
        self.arch1_cfg = [
            [[16, 16, 7, 2], [16, 32, 7, 1], [32, 32, 7, 1], [32, 24, 7, 1]],
            [[24, 64, 7, 2], [64, 64, 7, 1], [64, 64, 7, 1], [64, 64, 7, 1], [64, 64, 7, 1]],
            [[64, 64, 7, 2], [64, 64, 7, 1], [64, 72, 7, 1], [72, 72, 7, 1], [72, 72, 7, 1]],
            [[72, 72, 7, 1], [72, 80, 7, 1], [80, 100, 7, 1], [100, 120, 7, 1], [120, 140, 7, 1], [140, 160, 7, 1]]
        ]
        self.arch2_cfg = [
            [[16, 24, 7, 2], [24, 24, 7, 1], [24, 24, 7, 1], [24, 24, 7, 1], [24, 24, 7, 1], [24, 24, 7, 1]],
            [[24, 64, 7, 2], [64, 64, 7, 1], [64, 64, 7, 1], [64, 64, 7, 1], [64, 64, 7, 1], [64, 64, 7, 1], [64, 64, 7, 1], [64, 64, 7, 1]],
            [[64, 64, 7, 2], [64, 64, 7, 1], [64, 72, 7, 1], [72, 72, 7, 1], [72, 72, 7, 1], [72, 72, 7, 1], [72, 72, 7, 1], [72, 72, 7, 1]],
            [[72, 72, 7, 1], [72, 80, 7, 1], [80, 100, 7, 1], [100, 120, 7, 1], [120, 140, 7, 1], [140, 140, 7, 1], [140, 140, 7, 1], [140, 140, 7, 1], [140, 140, 7, 1], [140, 160, 7, 1]]
        ]
        self.deconvLayers_cfg = [
            [48, 24, 24], [4, 4, 4]
        ]

        # 特征准备过程，用于对齐骨干的输入特征大小。 -- Wu
        self.c1 = nn.Sequential(
            ConvBlockBase(3, 32, 3, 2),
            ConvBlockBase(32, in_channels, 3, 1)
        )

        # Backbone。根据Config中backbone的配置，使用lp_common_layers.py的ConvStage生成。 -- Wu
        backboneConf = self.arch2_cfg
        self.stages = []
        self.channels = [in_channels]

        for s in range(len(backboneConf)):
            # 特征提取的网络结构。 -- Wu
            self.stages.append(ConvStage(backboneConf[s]))
            # 记录所有卷积层的输出通道数，用于确定Head中的几个Deconv结构的输入、输出通道数。对应变小，也便于跟前面的特征进行残差融合。 -- Wu
            # 通道数由小变大，[16, 24, 64, 72, 160]，对应的特征尺寸由大变小。 -- Wu
            self.channels.append(backboneConf[s][-1][1])  # out_channels

        self.backbone = nn.ModuleList(self.stages)  # 最终生成好的backbone块。

        # Deconv Head
        self.loopLayers = []
        self.refineLayers = []
        self.refineChannels = self.channels[-1]      # -1: 160。
        deconvConf = self.deconvLayers_cfg
        for l in range(len(deconvConf)):  # 3
            rawChannels = self.channels[-l-2]        # -2(l=0): 72 | -3(l=1): 64 | -4(l=2): 24

            pad, out_pad = self.get_deconv_paddings(deconvConf[1][l])

            self.refineLayers.append(
                nn.ConvTranspose2d(
                    self.refineChannels,             # l=0: 72 | l=1: 48 | l=2: 24
                    deconvConf[0][l],                # l=0: 48 | l=1: 24 | l=2: 24
                    deconvConf[1][l],                # l=0: 4  | l=1: 4  | l=2: 4
                    2,
                    pad,
                    out_pad,
                    bias=False)
            )
            self.loopLayers.append(
                nn.ConvTranspose2d(
                    rawChannels,                     # l=0: 72 | l=1: 64 | l=2: 24
                    deconvConf[0][l],                # l=0: 48 | l=1: 24 | l=2: 24
                    deconvConf[1][l],                # l=0: 4  | l=1: 4  | l=2: 4
                    2,
                    pad,
                    out_pad,
                    bias=False)
            )
            self.refineChannels = deconvConf[0][l]  # l=0: 48 | l=1: 24 | l=2: 24

        self.loopLayers = nn.ModuleList(self.loopLayers)
        self.refineLayers = nn.ModuleList(self.refineLayers)

        # Output
        self.loopFinal = []
        self.refineFinal = []
        self.finalChannel = []
        for l in range(1, len(deconvConf[0])):  # 对后两个Deconv的输出进行预测。
            # 2*num_joints: num_joints channels represent heatmaps for each joint, the others num_joints channels are the tags
            # 2*num_joints：一个代表坐标点Heatmap。 -- Wu
            self.refineFinal.append(nn.Sequential(
                ConvBlockBase(deconvConf[0][l], deconvConf[0][l], 5),   # l=1: 24 | l=2: 24
                ConvBlockBase(deconvConf[0][l], self.kpsCount, 5)       # l=1: 24 | l=2: 24
            ))

            self.loopFinal.append(nn.Sequential(
                ConvBlockBase(self.channels[-l-3], self.channels[-l-3], 5),  # -4(l=1): 24 | -5(l=2): 16
                ConvBlockBase(self.channels[-l-3], self.kpsCount, 5)         # -4(l=1): 24 | -5(l=2): 16
            ))

        self.refineFinal = nn.ModuleList(self.refineFinal)
        self.loopFinal = nn.ModuleList(self.loopFinal)

        self.projection = self._get_projection(self.mode)

    def get_deconv_paddings(self, deconv_kernel):
        padding = 0
        output_padding = 0
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return padding, output_padding

    def forward(self, x):
        # Large Kernels Convs
        # 预处理，对齐。 -- Wu
        x = self.c1(x)

        # Backbone
        # 特征提取过程，保存c1和各Stage的输出特征。 -- Wu
        x_checkpoints = [x]  # 通道数：[16, 24, 64, 72, 160]
        for l in range(len(self.backbone)):
            x = self.stages[l](x)
            x_checkpoints.append(x)

        # 加映射

        # Deconv Head
        # 反卷积，预测过程。 -- Wu
        outputs = []
        for l in range(len(self.refineLayers)):
            x = self.refineLayers[l](x)  # 反卷积（类上采样）过程。 -- Wu
            x_loop = self.loopLayers[l](x_checkpoints[-l-2])  # 循环层（类SkipConnection）过程。 -- Wu
            x = x + x_loop  # 元素加（论文中的concatenate）

            # Final
            # 后两个Deconv过程进行结果预测。每个Deconv中，①分别进行对Refine和Loop输出特征的预测，前者相当于最终的预测，后者相当于对skip过来的原特征的预测；②将两个预测结果元素加（论文中的concatenate）
            if l > 0:
                finalForward = self.refineFinal[l-1](x)
                finalLoop = self.loopFinal[l-1](x_checkpoints[-l-3])
                outputs.append(finalForward+finalLoop)

        return outputs

    def _get_projection(self, mode):
        if mode == "MaxPool":
            return nn.Sequential(nn.MaxPool2d(2, 2))
        elif mode == "AvgPool":
            return nn.Sequential(nn.AvgPool2d(2, 2))
        elif self.mode == "ConvOne":
            return nn.Sequential(nn.Conv2d(128, 128, 1, 1, 0, bias=False))


def litePose(k, mode="default", nograd=False):
    model = LitePose(k, mode).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model