#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import Control_Exp1001 as CE
import os
import json
from collections import defaultdict


class PltUtil:
    def __init__(self, para= None, title = None):
        """
        专门用来画图的类
        :param para:
        :param title:
        """

        self.dic = defaultdict(list)

        if para is None:
            para = {}
        self.para = para
        self.title = title

    def push(self, name, value):
        self.dic[name].append(value)

    def plot(self):

        para = self.para
        plt.figure(**para)
        legend = []
        for key, values in self.dic.items():
            legend.append(key)
            plt.plot(values)
        plt.legend(legend)
        plt.show()




