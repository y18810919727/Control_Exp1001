#!/usr/bin/python
# -*- coding:utf8 -*-
import collections

import matplotlib
font = {'family': 'STSong',
        'size': 12}

matplotlib.rc("font", **font)
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

import numpy as np
import os

import copy

def max_absolute_error(a, b):
    res = np.max(np.abs(a-b))
    return res

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

    def zn2en(self, s):
        s=str(s)
        s = s.replace('浓度','Concentration')
        s = s.replace('进料','FeedIn')
        s = s.replace('高度','Height')
        s = s.replace('泵速','PumpSpeed')
        s = s.replace('设定值','SetPoint')
        s = s.replace('效用值','Utility')
        s = s.replace('底流','Underflow')
        s = s.replace('泥层','Slurry')
        s = s.replace('絮凝剂', 'Flocculant')
        s = s.replace('经验回放大小为', 'ReplaySize')
        s = s.replace('经验回放数量为', 'ReplaySize')
        s = s.replace('无经验回放', 'NoReplay')
        return s

    def extend_array(self, data, ratio=0.02):
        l = data.shape[0]
        extend_len = int(l*ratio)
        def tail_make(a, b, length):

            extend_array = []
            for i in range(length):
                if len(extend_array)>=2:
                    first = extend_array[-1]
                    second = extend_array[-2]
                elif len(extend_array) == 1:
                    first = extend_array[-1]
                    second = a
                else:
                    first = a
                    second = b
                extend_array.append(first*2 - second)
            return extend_array
        left_array = np.array(tail_make(data[0], data[1], extend_len))
        right_array = np.array(tail_make(data[-1], data[-2], extend_len))
        new_array = np.hstack((left_array, data, right_array))
        return new_array

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
                for col in ['orange', 'dodgerblue', 'green', 'red']:
                    yield col
            color_generator = color_list()
            for (key, values) in self.data.items():
                values_array = np.array(values)
                legend_name.append(key)
                line_color = 'k' if key=='设定值' else next(color_generator)
                x_array = np.arange(0, values_array.shape[0], 1)

                if key == '设定值':
                    plt.plot(self.extend_array(x_array/3),
                             self.extend_array(values_array[:, pic_id]), c=line_color,linewidth=2)
                else:
                    # 自动化学报要求黑白 为True代表画黑白图
                    automatical = False

                    if automatical:
                        if key =='HDP':
                            line_coefficient = 'k+-'
                        elif key =='DHP':
                            line_coefficient = 'kx-'
                        elif key =='ILPL':
                            line_coefficient = 'k^-'
                        elif key =='HCNVI':
                            line_coefficient = 'ko-'

                        if key =='HCNVI':
                            ms = 6
                        else:
                            ms = 3
                        radio = 10
                        x_index = np.arange(0, x_array.shape[0], 1)
                        x_index = (x_index*radio)[:(x_array.shape[0]//radio)]
                        plt.plot((x_array/3)[x_index], values_array[:, pic_id][x_index],line_coefficient, linewidth=1, ms=ms)
                    else:
                        plt.plot(x_array/3, values_array[:, pic_id], c=line_color, linewidth=2)
                    #plt.plot(x_array/3, values_array[:, pic_id], c=line_color, linewidth=2)

                # if key == 'HCNVI':
                #     plt.plot(x_array/3, values_array[:, pic_id], c=line_color,
                #              marker='o',markevery=80,ms=3)
                # else:
                #     plt.plot(x_array/3, values_array[:, pic_id], c=line_color)

            if not self.value_name[pic_id] == '进料浓度' \
                    and not self.value_name[pic_id] == '进料泵速':
                plt.legend(legend_name)

            plt.grid()
            plt.title(self.value_name[pic_id])
            if '浓' in self.value_name[pic_id]:
                plt.ylabel(r'$kg/m_3$')

            if '速' in self.value_name[pic_id]:
                plt.ylabel(r'$Hz$')
            plt.xlabel('时间(分钟)')
            from Control_Exp1001.demo.thickener_noise_chinese.common import exp_name
            img_root = os.path.join('../images/', exp_name.get_exp_name()) +'/'
            img_name = str(self.value_name[pic_id])+'_'.join(legend_name)
            img_name = img_name.replace(' ', '_')
            dest_name = self.zn2en(img_root + img_name + '.eps')
            plt.savefig(dest_name, format="eps", dpi=600)
            plt.show()

    def cal_mse(self):
        mse_dict={}
        if "设定值" not in self.data.keys():
            return
        for pic_id in range(1, self.size):
            set_point = np.array(self.data['设定值'])[:, pic_id]
            for (key, values) in self.data.items():
                if key == '设定值':
                    continue
                values_array = np.array(values)
                line_color = 'k' if key=='设定值' else None
                plt.plot(values_array[:, pic_id], c=line_color)
                print("MSE\t%s\t%f"%(key, mean_squared_error(
                    set_point, values_array[:, pic_id]
                )))

                print("IAE\t%s\t%f"%(key, mean_absolute_error(
                    set_point, values_array[:, pic_id]
                )))

                print("MAE\t%s\t%f"%(key, max_absolute_error(
                    set_point, values_array[:, pic_id]
                )))
                if self.value_name[pic_id] == '底流浓度':
                    mse_dict[key] = mean_squared_error(
                    set_point, values_array[:, pic_id]
                )
        return mse_dict



