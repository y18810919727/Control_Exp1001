#!/usr/bin/python
import sys
import os
import json
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
import sklearn.metrics.base
from sklearn.metrics import mean_squared_error

import torch.nn as nn
import torch.nn.functional as F
from Control_Exp1001.simulation.thickener import Thickener
from torch.autograd import Variable
import torch
import torch.optim as optim
from Control_Exp1001.control.base_ac import ACBase
from Control_Exp1001.demo.thickener.value_iterate import VI
from Control_Exp1001.demo.thickener.ILPL.critic import Critic
from Control_Exp1001.demo.thickener.ILPL.actor import Actor
from Control_Exp1001.demo.thickener.ILPL.predict import Model
sys.path.append(('./'))
import itertools
from Control_Exp1001.demo.flotation.plotuilt import PltUtil
import  mpl_toolkits.mplot3d as p3d
from pylab import contourf
from pylab import contour



class ViSample(VI):
    def __init__(self,
                 **para,
                 ):

        super(ViSample, self).__init__(**para)

    def data_generator(self, train_x, train_y):
        n = train_x.shape[0]
        tmp_x = np.copy(train_x).squeeze()[:,4:6]
        mean = np.mean(tmp_x, axis=0)
        cov = np.cov(tmp_x.T)
        from scipy.stats import multivariate_normal
        rv = multivariate_normal(mean=mean, cov=cov)
        data_p = [1.0/rv.pdf(x) for x in tmp_x]
        data_p_sum = np.sum(data_p)
        data_p = list(map(lambda x:x/data_p_sum, data_p))
        for _ in range(n):
            index = np.random.choice(n, self.predict_batch_size, data_p)
            yield train_x[index,:,:], train_y[index,:,:]


    def train_identification_model(self):
        """

        代码参考:
        https://www.pytorchtutorial.com/3-6-optimizer/#i
        :param Vm:
        :param Lm:
        :return:
        """
        # 定义预测模型
        self.model_criterion = torch.nn.MSELoss()

        train_x, train_y = self.cal_training_data()
        train_x, train_y = self.normalize(train_x, train_y, new_para=True)
        # plt.plot(train_y[:,0, 0])
        # plt.show()
        # plt.plot(train_y[:,0, 1])
        # plt.show()


        mse_list = []
        # 训练预测模型
        for epoch in itertools.count():

            print("Epoch:{}".format(epoch+1), "")
            sum_loss = 0
            for step,(data_x, data_y) in enumerate(self.data_generator(train_x, train_y)):
                batch_x = torch.FloatTensor(data_x)
                batch_y = torch.FloatTensor(data_y)
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                output = self.model_nn(b_x)
                loss = self.model_criterion(output, b_y)
                self.model_nn_optim.zero_grad()
                loss.backward()

                self.model_nn_optim.step()
                self.predict_training_losses.append(loss.item())
                sum_loss += loss.item()


            print("Loss:{}".format(sum_loss))
            # 每过20轮评估一次mse
            # if epoch % 20 == 0:
            #     mse_list.append(self.cal_predict_mse())
            # loss足够小或者迭代次数超过50次结束
            if sum_loss < self.model_nn_error_limit or epoch >= self.predict_epoch:
                break



        # # 最后再评估一次
        # mse_list.append(self.cal_predict_mse())
        # # 绘制损失变化
        # plt.figure()
        #
        # plt.title("Loss in various epoch")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.plot(self.predict_training_losses)
        # plt.show()
        #
        #
        # # 绘制预测mse变化
        # plt.figure()
        # mse_array = np.array(mse_list)
        # for i in range(mse_array.shape[1]):
        #     plt.plot(mse_array[:,i])
        #     plt.plot(mse_array[:,i])
        # plt.legend(['y1','y2'])
        # plt.show()
        # # 打印mse
        # print(mse_array)

