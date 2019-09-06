#!/usr/bin/python
# -*- coding:utf8 -*-
import collections

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import os

import copy


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
        para = copy.deepcopy(self.para)
        fig = plt.figure(**para)
        for pic_id in range(self.size):
            legend_name = []

            def color_list():
                for col in ['y', 'purple', 'blue', 'red']:
                    yield col
            color_generator = color_list()
            for (key, values) in self.data.items():
                values_array = np.array(values)
                legend_name.append(key)
                line_color = 'k' if key=='set point' else next(color_generator)
                x_array = np.arange(0, values_array.shape[0], 1)

                if line_color is 'k':
                    plt.plot(x_array/3, values_array[:, pic_id], c=line_color,linewidth=4)
                else:
                    plt.plot(x_array/3, values_array[:, pic_id], c=line_color, linewidth=1)

            if not self.value_name[pic_id] == 'Concentration(In)' \
                    and not self.value_name[pic_id] == 'pulp speed(Feed In)':
                plt.legend(legend_name)

            plt.title(self.value_name[pic_id])
            plt.xlabel('time(minute)')
            from Control_Exp1001.demo.thickener.common import exp_name
            img_root = os.path.join('../images/', exp_name.get_exp_name()) +'/'
            img_name = str(self.value_name[pic_id])+'_'.join(legend_name)
            img_name = img_name.replace(' ', '_')
            plt.savefig(img_root + img_name + '.png', dpi=300)
            plt.show()

    def cal_mse(self):
        mse_dict={}
        if "set point" not in self.data.keys():
            return
        for pic_id in range(1, self.size):
            set_point = np.array(self.data['set point'])[:, pic_id]
            for (key, values) in self.data.items():
                if key == 'set point':
                    continue
                values_array = np.array(values)
                line_color = 'k' if key=='set point' else None
                plt.plot(values_array[:, pic_id], c=line_color)
                print("MSE\t%s\t%f"%(key, mean_squared_error(
                    set_point, values_array[:, pic_id]
                )))
                if self.value_name[pic_id] == 'Concentration(out)':
                    mse_dict[key] = mean_squared_error(
                    set_point, values_array[:, pic_id]
                )
        return mse_dict



