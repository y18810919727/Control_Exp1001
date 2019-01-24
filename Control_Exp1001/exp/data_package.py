#!/usr/bin/python
# -*- coding:utf8 -*-
import collections

import matplotlib.pyplot as plt
import numpy as np


class DataPackage:
    def __init__(self, exp_name=None, value_name=None, para=None):
        """

        :param exp_name: 实验名称
        :param value_name: 数据每个维度的名称——列表
        :param para: 画图参数
        """
        if exp_name is None:
            raise ValueError('exp_name should not be none')
        self.exp_name = exp_name
        self.value_name = value_name
        self.size = None
        self.data = collections.defaultdict(list)
        if para is None:
            para = {}
        self.para = para

    def push(self, x, name=None):
        """

        :param x: shape(1,x)
        :param name:
        :return:
        """
        value = np.array(x).reshape(-1)
        if self.size is None:
            self.size = value.shape[0]
        else:
            if self.size != value.shape[0]:
                raise ValueError("Dimensional inconsistency! of DataPackage %s", self.exp_name)
        if name is None:
            self.data[self.exp_name].append(value)
        else:
            self.data[name].append(value)

    # 和其他DataPackage合并
    def merge(self, dp):
        if not isinstance(dp, DataPackage):
            raise ValueError('merged object should be an instance of DataPackage')
        for (key, values) in dp.data.items():
            self.data[key] = values

    def merge_list(self, dp_list):
        for dp in dp_list:
            self.merge(dp)
        return self

    def plt(self):
        if self.value_name is None:
            self.value_name = [str(i) for i in range(self.size)]

        if self.size == 0:
            return
        if len(self.value_name) != self.size:
            raise ValueError('size of value_name and size are not match')
        para = self.para
        fig = plt.figure(**para)
        for pic_id in range(self.size):
            legend_name = []
            for (key, values) in self.data.items():
                values_array = np.array(values)
                legend_name.append(key)
                plt.plot(values_array[:, pic_id])
            plt.legend(legend_name)

            plt.title(self.value_name[pic_id])
            plt.show()
