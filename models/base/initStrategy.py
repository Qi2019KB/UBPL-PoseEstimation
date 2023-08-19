# -*- coding: utf-8 -*-
import math
import torch.nn as nn


class InitializeStrategy:
    def __init__(self):
        pass

    # region function：模型参数初始化
    # params：
    #   （1）net：模型对象
    #   （2）mode：模式 ["constant", "uniform", "normal"，"xavier", "kaiming_normal", "kaiming_uniform"]
    # endregion
    @classmethod
    def parameters_initialize(cls, net, mode):
        # nn.init.uniform_(tensor, a=0., b=1.)  # 对张量赋值-均匀分布，默认取值范围(0., 1.)
        # nn.init.constant_(tensor, val)  # 对张量赋值-常量，需要赋一个常数值
        # nn.init.normal_(tensor, mean=0, std=1)  # 对张量赋值-高斯（正太）分布。mean均值，std方差
        # nn.init.xavier_uniform_(tensor, gain=nn.init.calculate_gain("relu"))  # 对张量赋值-xavier初始化。gain可根据激活函数种类获得：nn.init.calculate_gain("relu")
        # nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')  # 对张量赋值-kaiming初始化
        # nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')  # 对张量赋值-kaiming正太分布初始化
        for layer in net.modules():
            if mode == "constant":
                if isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.weight, 0.)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal(layer.weight, std=1e-3)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            elif mode == "uniform":
                if isinstance(layer, nn.Conv2d):
                    n = layer.in_channels
                    for k in layer.kernel_size: n *= k
                    stdv = 1. / math.sqrt(n)
                    nn.init.uniform_(layer.weight, -stdv, stdv)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -stdv, stdv)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.uniform_(layer.weight)
                    nn.init.uniform_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    n = layer.in_channels
                    stdv = 1. / math.sqrt(n)
                    nn.init.uniform_(layer.weight, -stdv, stdv)
                    if layer.bias is not None:
                        nn.init.uniform_(layer.bias, -stdv, stdv)
            elif mode == "normal":
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.normal_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.normal_(layer.weight)
                    nn.init.normal_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.normal_(layer.bias)
            elif mode == "xavier":
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal(layer.weight, std=1e-3)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            elif mode == "kaiming_normal":
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.reset_running_stats()
                    if layer.affine:
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
            #  mode == "kaiming_uniform"
            else:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.reset_running_stats()
                    if layer.affine:
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)
