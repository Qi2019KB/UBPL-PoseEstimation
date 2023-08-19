# -*- coding: utf-8 -*-
import os

# 全局缓存对象
global glob_cache
glob_cache = {}


def getValue(key, default=None):
    try:
        return glob_cache[key]
    except KeyError:
        return default


def setValue(key, value):
    glob_cache[key] = value
    return value


# 项目级信息（勿动）
root = os.path.abspath(os.path.dirname(__file__))  # 项目根路径
project = root.split("\\")[-1]  # 项目名称

# 实验级配置项
expr = "E:/00Experiment/expr"  # 实验数据存储位置
temp = "E:/00Experiment/temp"  # 临时数据存储位置
stat = "E:/00Experiment/statistic"  # 统计数据存储位置
