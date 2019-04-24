#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json

import os
import matplotlib
from matplotlib import pyplot as plt


font = {'family': 'STSong',
        'size': 12}

matplotlib.rc("font", **font)
matplotlib.style.use('ggplot')
path = os.path.join('images', 'VIandHDP', 'time_cmp.eps')
def time_generate():

    time_used = {}
    for line in open('time_cmp_data', 'r').readlines():
        item_time = float(line.split('-')[2].strip())
        if 'MSE' in line:
            break
        yield item_time


time_used = {}
time_used['HDP']=([],[],[])
time_used['HCNVI']=([],[],[])
time_ge = time_generate()
for round in range(10):
    for exp_name in ['HDP', 'HCNVI']:
        for id, time_type in enumerate(['总时间','训练累计时间', '计算控制动作时间']):
            cur_time = time_ge.__next__()
            time_used[exp_name][id].append(cur_time)

def plt_bar(data):
    plt.figure(111)
    left = np.arange(0,20,2)
    dis = 0.5

    for key, values in data.items():
        plt.bar(left+dis, values[2],width=0.5, bottom=values[1])
        plt.bar(left+dis, values[1],width=0.5)
        dis+=0.5
    plt.ylabel('时间/s')
    plt.xlabel('实验轮次')
    x_name_list = list(map(str, np.arange(1,11,1)))
    plt.xticks(left+0.75,x_name_list)
    #plt.xticks(left+1.75, ('G1', 'G2', 'G3', 'G4', 'G5'))

    plt.legend(['HDP计算控制输入','HDP训练','HCNVI计算控制输入','HCNVI训练'])
    #plt.title("时间对比")
    plt.savefig(path, dpi=600,format='eps')
    plt.show()





print(time_used)
plt_bar(time_used)




