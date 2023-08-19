# -*- coding: utf-8 -*-
import time
import datetime
import os
import GLOB as glob


# region function：日志类
# description：用于分级显示、存储log日志。
# params：
#   （1）experiment：实验名称
#   （2）levelParams：级别参数 [{"levelName": 级别名称, "threshold": 级别阈值, "filename": 对应文件名称}]
#   （3）dataCount：数据集数量
#   （4）rebuild：是否重新生成数据集（不使用Storage预存的版本）
# return：标签数据集
# endregion
class Logger:
    _default_levelConfigs = [
        {"level": "L3", "threshold": 80, "filename": "log_L3.log"},
        {"level": "L2", "threshold": 90, "filename": "log_L2.log"},
        {"level": "L1", "threshold": 100, "filename": "log_L1.log"}
    ]

    def __init__(self, experiment, consoleLevel, levelConfigs=None):
        self.levelConfigs = self._default_levelConfigs if levelConfigs is None else levelConfigs
        self.loggers = [self._logger(experiment, levelParams) for levelParams in self.levelConfigs]
        self.consoleLevel = consoleLevel

    def print(self, level, content, start=None, end=None):
        content = self._format_level(level, self._format_time(content, start, end))
        if self._checkConsole(level): print(content)
        for logger in self._getEnableLogger(level):
            loggerObj = logger["logger"]
            loggerObj.write(content+"\n")
            loggerObj.flush()
        time.sleep(0.01)

    def _logger(self, experiment, levelParams):
        pathname = "{}/{}/logs/{}".format(glob.expr, experiment, levelParams["filename"])
        # 创建路径
        folderPath = os.path.split(pathname)[0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        logger = open(pathname, 'a+')
        return {"level": levelParams["level"], "threshold": levelParams["threshold"], "logger": logger}

    def _format_time(self, content, start=None, end=None):
        now = end if end is not None else datetime.datetime.now()
        # interval = self._interval_format(seconds=(now - now).seconds) if start is None else self._interval_format(seconds=(now - start).seconds)
        interval = "-" if start is None else self._interval_format(seconds=(now - start).seconds)
        return "{} ({}): {}".format(now.strftime("%m-%d %H:%M"), interval, content)

    def _format_level(self, level, content):
        return "[{}] {}".format(level, content)

    def _getEnableLogger(self, level):
        loggerObjArray = []
        for logger in self.loggers:
            if logger["threshold"] <= self._getThreshold(level):
                loggerObjArray.append(logger)
        return loggerObjArray

    def _checkConsole(self, level):
        return self._getThreshold(level) >= self._getThreshold(self.consoleLevel)

    def _getThreshold(self, level):
        return [levelParams for levelParams in self.levelConfigs if levelParams["level"] == level][0]["threshold"]

    def _interval_format(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)

