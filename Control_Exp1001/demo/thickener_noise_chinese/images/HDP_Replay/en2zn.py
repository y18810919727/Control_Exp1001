#!/usr/bin/python
# -*- coding:utf8 -*-
import os
def name_trans(name):
    s = str(name)
    s = s.replace('Concentration','浓度')
    s = s.replace('FeedIn','进料')
    s = s.replace('Height','高度')
    s = s.replace('PumpSpeed','泵速')
    s = s.replace('SetPoint','设定值')
    s = s.replace('Utility','效用值')
    s = s.replace('Underflow','底流')
    s = s.replace('Slurry','泥层')
    s = s.replace('Flocculant','絮凝剂' )
    s = s.replace('ReplaySize','经验回放数量为')
    s = s.replace('NoReplay','无经验回放')
    s = s.replace('ReplaySize','经验回放数量为')
    s = s.replace('NoReplay','无经验回放')
    return s
for file in os.listdir('.'):
    print(file)
    file_name = os.path.join('.', file)
    os.rename(file_name, name_trans(file_name))
