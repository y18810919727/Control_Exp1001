#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

plt.figure()
param = {
    ''
}
a = np.arange(0,160, 0.1)
plt.plot(a,'-',marker='o',markevery=80,c='blue',scalex=10, linewidth=1)

plt.plot(a/2,'-',marker='o',markevery=80,c='y',scalex=0.1,ms=5)
plt.legend(['10', '1'])
#plt.scatter(np.arange(a.shape[0]/100)*100, b)
plt.show()

