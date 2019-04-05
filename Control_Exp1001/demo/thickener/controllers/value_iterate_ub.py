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
from Control_Exp1001.demo.thickener.controllers.value_iterate import VI
from Control_Exp1001.demo.thickener.ILPL.critic import Critic
from Control_Exp1001.demo.thickener.ILPL.actor import Actor
from Control_Exp1001.demo.thickener.ILPL.predict import Model
sys.path.append(('./'))
import itertools
from Control_Exp1001.demo.flotation.plotuilt import PltUtil
import  mpl_toolkits.mplot3d as p3d
from pylab import contourf
from pylab import contour



class VIub(VI):
    def __init__(self,
                 find_time_max=50,
                 find_lr=0.4,
                 **para,
                 ):

        super(VIub, self).__init__(**para)
        self.find_time_max =find_time_max
        self.find_lr = find_lr
        self.last_act = None

    def _act(self, state):

        y = self.normalize_y(state[self.indice_y])
        y_star = self.normalize_y(state[self.indice_y_star])
        c = self.normalize_c(state[self.indice_c])

        y = torch.FloatTensor(y).unsqueeze(0)
        y_star = torch.FloatTensor(y_star).unsqueeze(0)
        c = torch.FloatTensor(c).unsqueeze(0)

        self.delta_u = self.env.u_bounds[:, 1] - self.env.u_bounds[:, 0]
        self.mid_u = np.mean(self.env.u_bounds, axis=1)
        x = torch.FloatTensor(np.hstack((y, y_star,c))).unsqueeze(0)
        #print('#' * 20)

        # region 基于python scipy.optimize.minimize 求最优act
        # 可能由于fun太过复杂，最后求得得x几乎与x0相同，该方案阉割
        # def fun(x):
        #     return x[0]*x[0]+ x[1]*x[1]
        #     act = torch.FloatTensor([x[0], x[1]]).unsqueeze(0)
        #     y_pred = self.model_nn(torch.cat((y, act, c), dim=1))
        #     diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
        #     det_u = torch.nn.functional.linear(input=act, weight=torch.diag(diff_U/2))
        #     penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
        #         det_u.t()
        #     )).diag().unsqueeze(dim=1)
        #
        #     J_pred = self.critic_nn(torch.cat((y_pred, y_star, c), dim=1))
        #     #penalty_u = torch.zeros(J_pred.shape)
        #     J_loss = penalty_u + self.gamma * J_pred
        #     return 1000 * float(J_loss)
        #
        # bnds = ((-1,1),(-1,1))
        # act_begin =np.array([
        #     np.random.uniform(-1,1),
        #     np.random.uniform(-1,1)
        #     ]
        # )
        # res = minimize(fun=fun, x0=act_begin, bounds=bnds, method='SLSQP')
        # act = res.x
        # endregion

        #region 基于梯度下降方法求act = argmin(J(act))
        # ########### 随机选择act起点 ##########
        # act = torch.rand((1,self.env.size_yudc[1]))
        #
        # ########### 以上一个step的action作为起点 ###
        # if self.step == 0:
        #     act = torch.rand((1,self.env.size_yudc[1]))
        # else:
        #     act = torch.FloatTensor(self.last_act)
        # lr = 10
        # gamma = 0.9
        #
        # act.requires_grad = True
        # last_loss = np.inf
        # loop_time = 0
        #
        # loop = 1000
        # while loop>0:
        #     loop -= 1
        #
        #     y_pred = self.model_nn(torch.cat((y, act, c), dim=1))
        #     U = torch.FloatTensor(self.mid_u)
        #     diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
        #     det_u = torch.nn.functional.linear(input=act, weight=torch.diag(diff_U/2))
        #     penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
        #         det_u.t()
        #     )).diag().unsqueeze(dim=1)
        #
        #     J_pred = self.critic_nn(torch.cat((y_pred, y_star, c), dim=1))
        #     #penalty_u = torch.zeros(J_pred.shape)
        #     J_loss = penalty_u + self.gamma * J_pred
        #     # J_loss = penalty_u
        #     J_loss = J_loss.mean()
        #     J_loss.backward()
        #     grad = act.grad
        #
        #
        #     ### 三分法寻找最佳lr, 由于速度太慢已经放弃已经放弃 ###
        #     # def fun(lr):
        #     #     tmp_act = act.clone()
        #     #     tmp_act.data -= lr * grad
        #     #     y_pred = self.model_nn(torch.cat((y, tmp_act, c), dim=1))
        #     #     U = torch.FloatTensor(self.mid_u)
        #     #     diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
        #     #     det_u = torch.nn.functional.linear(input=act, weight=torch.diag(diff_U/2))
        #     #     penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
        #     #         det_u.t()
        #     #     )).diag().unsqueeze(dim=1)
        #     #
        #     #     J_pred = self.critic_nn(torch.cat((y_pred, y_star, c), dim=1))
        #     #     #penalty_u = torch.zeros(J_pred.shape)
        #     #     J_loss = penalty_u + self.gamma * J_pred
        #     #     return float(J_loss)
        #     #
        #     #
        #     # def find_best_lr(L, R, fun):
        #     #     return 10
        #     #     if L+(1e-1)>R:
        #     #         return (L+R)/2
        #     #     v2 = fun(2*L/3.0+1*R/3)
        #     #     v3 = fun(1*L/3.0+2*R/3)
        #     #     if v2>v3:
        #     #         return find_best_lr(2*L/3.0+1*R/3, R, fun)
        #     #     else:
        #     #         return find_best_lr(L, 1*L/3.0+2*R/3, fun)
        #
        #     # lr_max = float(torch.min((torch.abs(torch.ones(grad.shape)/grad))))
        #     # lr = find_best_lr(0, min(lr_max/4, 30), fun=fun)
        #     #####################################################
        #
        #     lr = 10
        #
        #     #print("Loss\t%f\tmax\t%f\tlr\t%f"%(J_loss, lr_max, lr))
        #
        #     act.data = act.data - lr*grad
        #     act.grad = torch.zeros(act.shape)
        #     print('loop:', loop_time)
        #     print('grad:', grad)
        #     print('act:', act)
        #     if torch.max(torch.abs(act)) >= 1:
        #         act.data = act.data + lr*grad
        #         break
        #     if float(J_loss)<1e-3:
        #         break
        #     # if J_loss.data+(1e-8) > last_loss:
        #     #     lr = lr*gamma
        #     # elif J_loss>0 and 0 < (last_loss-J_loss)/J_loss < 0.1:
        #     #     lr = lr / gamma
        #     #     #print("intermediate lr",lr)
        #     # last_loss = float(J_loss.data)
        #     # loop_time += 1
        #     # # if loop_time>10000:
        #     # #     raise ValueError("can't find act")
        #     # if loop_time >1000:
        #     #     break
        # # print("final lr", lr)
        # print("find time :", loop_time)

        # act = act.detach().numpy()
        # endregion

        # region 使用torch自带优化器来求解最优act


        if self.last_act is None:
            act = torch.nn.Parameter(torch.FloatTensor([[0, 0]]))
        else:
            act = torch.nn.Parameter(torch.FloatTensor(self.last_act))

        opt = torch.optim.SGD(params=[act], lr=self.find_lr)

        act_begin = np.copy(act.data.numpy()).squeeze()
        act_list = []
        act_list.append(np.copy(act.data.numpy()).squeeze())
        find_time = 0
        while True:
            if find_time > self.find_time_max:
                break
            find_time += 1
            old_act = act.clone()
            diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
            det_u = torch.nn.functional.linear(input=act, weight=torch.diag(diff_U/2))
            # penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
            #     det_u.t()
            # )).diag().unsqueeze(dim=1)
            penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
                det_u.t()
            )).diag()

            y.requires_grad=True
            y_pred = self.model_nn(torch.cat((y, act, c), dim=1))
            J_pred = self.critic_nn(torch.cat((y_pred, y_star, c), dim=1))
            #penalty_u = torch.zeros(J_pred.shape)
            J_loss = penalty_u + self.gamma * J_pred
            J_loss = J_loss.mean()
            opt.zero_grad()
            J_loss.backward()
            opt.step()
            act.data = torch.nn.Parameter(torch.clamp(act,min=-1,max=1)).data
            act_list.append(np.copy(act.data.numpy()).squeeze())
            self.u_iter_times += 1
            find_time += 1
            if torch.dist(act, old_act)<1e-4:
                break

        act = act.detach().numpy()

        # endregion
        #print("final act:", act)
        #print('#'*20)


        self.last_act = np.copy(act)
        # print(act)
        # make the output action locate in bounds of constraint
        # U = (max - min)/2 * u + (max + min)/2



        A = np.matrix(np.diag(self.delta_u/2))
        B = np.matrix(self.mid_u).T
        act = A*np.matrix(act).T + B
        act = np.array(act).reshape(-1)
        # self.actor_nn[-1].weight.data = torch.FloatTensor()
        # self.actor_nn[-1].bias.data = torch.FloatTensor(self.mid_u)
        # self.actor_nn[-1].weight.requires_grad = False
        # self.actor_nn[-1].bias.requires_grad = False

        # if (self.step-1) % 20  == 0:
        #     self.test_critic_nn(title='round:'+str(self.step), cur_state=y, act_list=act_list)
        # if 195<self.step-1<220:
        #     self.test_critic_nn(title='round:'+str(self.step), cur_state=y, act_list=act_list)

        return act

