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
from Control_Exp1001.demo.thickener.ILPL.critic import Critic
from Control_Exp1001.demo.thickener.ILPL.actor import Actor
from Control_Exp1001.demo.thickener.ILPL.predict import Model
sys.path.append(('./'))
import itertools
from Control_Exp1001.demo.flotation.plotuilt import PltUtil
import  mpl_toolkits.mplot3d as p3d
from pylab import contourf
from pylab import contour


class ADHDP(ACBase):
    def __init__(self,

                 gpu_id=1,
                 replay_buffer = None,
                 u_bounds = None,
                 exploration = None,
                 env=None,
                 gamma=0.6,

                 batch_size = 1,
                 predict_batch_size=32,

                 critic_nn_error_limit = 1,
                 actor_nn_error_limit = 0.1,

                 actor_nn_lr = 0.01,
                 critic_nn_lr = 0.01,

                 indice_y = None,
                 indice_u = None,
                 indice_y_star = None,
                 indice_c=None,
                 hidden_critic = 10,
                 hidden_actor = 10,
                 max_iter_c= 30,
                 off_policy = False
                 ):
        """

        :param gpu_id:
        :param replay_buffer:
        :param u_bounds:
        :param exploration:
        :param env:
        :param predict_training_rounds:  训练预测模型时使用的真实数据条数
        :param Vm:
        :param Lm:
        :param Va:
        :param La:
        :param Vc:
        :param Lc:
        :param gamma:
        :param batch_size:
        :param predict_batch_size: 训练预测模型时的batch_size
        :param model_nn_error_limit:
        :param critic_nn_error_limit:  critic网络的误差限
        :param actor_nn_loss:
        :param u_iter: 求解u*时的迭代次数
        :param u_begin: 求解u*时，第一次迭代的其实u(k)
        :param indice_y: y在state中的位置
        :param indice_y_star: *在state中的位置
        :param u_first: 第一次控制时的命令
        """
        super(ADHDP, self).__init__(gpu_id=gpu_id,replay_buffer=replay_buffer,
                                   u_bounds=u_bounds,exploration=exploration)
        if env is None:
            env = Thickener()

        self.env=env

        self.device = None
        self.cuda_device(gpu_id)
        self.batch_size = batch_size



        self.critic_nn_error_limit = critic_nn_error_limit
        self.actor_nn_error_limit = actor_nn_error_limit


        dim_c = env.size_yudc[3]
        dim_y = env.size_yudc[0]
        dim_u = env.size_yudc[1]


        #定义actor网络相关
        self.actor_nn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(2*dim_y+dim_c, dim_u,bias=False),
            nn.Tanh(),
           # nn.Linear(dim_u, dim_u)
        )

        self.actor_nn_optim = torch.optim.SGD(self.actor_nn.parameters(), lr=actor_nn_lr)



        #定义critic网络相关:HDP

        self.critic_nn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(dim_y+dim_y+dim_c+dim_u, 1,bias=False),
        )
        self.critic_nn_optim = torch.optim.SGD(self.critic_nn.parameters(), lr=critic_nn_lr)
        self.critic_criterion = torch.nn.MSELoss()


        self.gamma = gamma

        if indice_y is None:
            indice_y = [2,3]
        if indice_y_star is None:
            indice_y_star = [0,1]
        if indice_u is None:
            indice_u = [4,5]
        self.indice_y = indice_y
        self.indice_y_star = indice_y_star
        self.indice_c = [6, 7]
        self.indice_u = indice_u
        self.max_iter_c = max_iter_c
        self.off_policy = off_policy

    def cuda_device(self, cuda_id):
        use_cuda = torch.cuda.is_available()
        cuda = 'cuda:'+str(cuda_id)
        self.device = torch.device(cuda if use_cuda else "cpu")

    def _act(self, state):

        y = self.normalize_y(state[self.indice_y])
        y_star = self.normalize_y(state[self.indice_y_star])
        c = self.normalize_c(state[self.indice_c])

        x = torch.FloatTensor(np.hstack((y, y_star,c))).unsqueeze(0)
        act = self.actor_nn(x).detach().squeeze(0).numpy()
        print('_act',act)
        # make the output action locate in bounds of constraint
        # U = (max - min)/2 * u + (max + min)/2

        self.delta_u = self.env.u_bounds[:, 1] - self.env.u_bounds[:, 0]
        self.mid_u = np.mean(self.env.u_bounds, axis=1)

        A = np.matrix(np.diag(self.delta_u/2))
        B = np.matrix(self.mid_u).T
        act = A*np.matrix(act).T + B
        act = np.array(act).reshape(-1)
        # self.actor_nn[-1].weight.data = torch.FloatTensor()
        # self.actor_nn[-1].bias.data = torch.FloatTensor(self.mid_u)
        # self.actor_nn[-1].weight.requires_grad = False
        # self.actor_nn[-1].bias.requires_grad = False

        return act


    def best_action(self, ny, y_star, nc):

        begin_act = self.actor_nn(torch.cat([ny,y_star,nc],dim=1))

        #next_q_value = self.critic_nn(torch.cat((ny, y_star, nc, next_action), dim=1))

        act = torch.nn.Parameter(begin_act)
        opt = torch.optim.SGD(params=[act], lr=1.0)

        act_list = []
        act_list.append(np.copy(act.data.numpy()).squeeze())
        find_loop_time=0
        while True:
            find_loop_time += 1
            old_act = act.clone()

            Q_loss = self.critic_nn(torch.cat((ny, y_star, nc, act), dim=1))
            #penalty_u = torch.zeros(J_pred.shape)
            opt.zero_grad()
            Q_loss = Q_loss.mean()
            Q_loss.backward()
            opt.step()
            act.data = torch.nn.Parameter(torch.clamp(act,min=-1,max=1)).data
            act_list.append(np.copy(act.data.numpy()).squeeze())
            if torch.dist(act, old_act)<1e-4:
                break
        print("find best action loop:", find_loop_time)
        return act

    def _train(self, s, u, ns, r, done):

        # 先放回放池
        self.replay_buffer.push(s, u, r, ns, done)
        # if len(self.replay_buffer) < self.batch_size:
        #     return
        # 从回放池取数据，默认1条
        state, action, reward, next_state, done = self.replay_buffer.sample(
            # 尽快开始训练，而不能等batchsize满了再开始
            min(len(self.replay_buffer), self.batch_size)
        )

        # 更新模型
        self.update_model(state, action, reward, next_state, done)

    def update_model(self,state, action, penalty, next_state, done):

        tmp_state = np.copy(state)
        state = torch.FloatTensor(self.normalize_state(state)).to(self.device)
        next_state = torch.FloatTensor(self.normalize_state(next_state)).to(self.device)
        action = torch.FloatTensor(self.normalize_u(action)).to(self.device)
        penalty = torch.FloatTensor(penalty).unsqueeze(1).to(self.device)
        indices_y = torch.LongTensor(self.indice_y)
        indices_c = torch.LongTensor(self.indice_c)
        indices_y_star = torch.LongTensor(self.indice_y_star)
        y = torch.index_select(state, 1, indices_y)

        ny = torch.index_select(next_state, 1, indices_y)
        y_star = torch.index_select(state, 1, indices_y_star)

        c = torch.index_select(state, 1, indices_c)
        nc = torch.index_select(next_state, 1, indices_c)

        # if (self.step-1) % 30 ==0:
        #     self.test_critic_nn(title='round:'+str(self.step))


        # region update model nn
        # while True:
        #
        #     next_state_predict = self.model_nn(torch.cat((y, action, c), dim=1))
        #     model_loss = self.model_criterion(ny, next_state_predict)
        #     self.model_nn_optim.zero_grad()
        #     model_loss.backward()
        #     self.model_nn_optim.step()
        #     # The loop will be teiminated while the average loss < limit
        #     if model_loss.data / self.batch_size < self.model_nn_error_limit:
        #         break
        # endregion


        # 循环更新actor网络和critic网路
        loop_time = 0
        last_J = np.inf
        # region update critic nn
        loop_time = 0
        while True:
            q_value = self.critic_nn(torch.cat((y, y_star, c, action), dim=1))

            if self.off_policy:
                next_action = self.best_action(ny, y_star, nc)
            else:
                next_action = self.actor_nn(torch.cat([ny,y_star,nc],dim=1))
            next_q_value = self.critic_nn(torch.cat((ny, y_star, nc, next_action), dim=1))
            target_q = penalty + self.gamma * next_q_value
            #print(target_q)

            # 定义TD loss
            critic_loss = self.critic_criterion(q_value, Variable(target_q.data))

            # critic_loss = self.critic_criterion(q_value, target_q)
            # critic_loss = self.critic_criterion(q_value, target_q.detach())

            next_q_value.required_grad = True
            next_q_value.register_hook(lambda grad:print(grad))
            self.critic_nn_optim.zero_grad()
            critic_loss.backward()
            self.critic_nn_optim.step()
            loop_time += 1
            if critic_loss < self.critic_nn_error_limit:
                break
            if loop_time >= self.max_iter_c:
                break

        # endregion
        print('step:',self.step, 'critic loop',loop_time)
        act_list2 = action.clone().detach().numpy()
        #print('train_critic', act_list2[-1])
        self.draw_uk_Jk(
            y=y[-1:, ],
            y_star=y_star[-1:, ],
            c=c[-1:, ],
            nc=nc[-1:, ],
            title='critic-Rounds-' + str(self.step),
            act_list=act_list2
        )



        if self.step%1 == 1:
            for param_group in self.actor_nn_optim.param_groups:
                param_group['lr'] *= 1
            act_list = []
            loop_time = 0
            while True:
                # region update actor nn
                action = self.actor_nn(torch.cat([ny,y_star,nc],dim=1))
                act_list.append(action.clone().detach().numpy()[-1])
                # J(k+1) = U(k)+J(y(k+1),c)
                J_pred = self.critic_nn(torch.cat((ny, y_star, nc, action), dim=1))
                J_loss = J_pred.mean()

                #action.requires_grad = True
                global u_grad
                action.register_hook(self.u_grad_cal)

                self.actor_nn_optim.zero_grad()
                J_loss.backward()
                self.actor_nn_optim.step()

                #print('critic loss', critic_loss)`
                loop_time += 1
                # if abs(J_loss-last_J) < self.actor_nn_error_limit:
                #     break
                last_J = float(J_loss)
                if J_loss < 1e-4:
                    break
                if loop_time > 100:
                    break
                # endregion

            self.draw_uk_Jk(
                y=y[-1:,],
                y_star=y_star[-1:,],
                c=c[-1:,],
                nc=nc[-1:,],
                title='Rounds-'+str(self.step),
                act_list=act_list
            )
            #print('train_actor', act_list[-1])



        #print('step:',self.step, 'act loop',loop_time)

    def u_grad_cal(self, grad):
        global u_grad
        u_grad = grad
    def y_grad_cal(self, grad):
        global y_pred_grad
        y_pred_grad = grad



    def normalize_u(self, u):
        u_max = self.u_bounds[:, 1]
        u_min = self.u_bounds[:, 0]
        new_u = 2*(u- u_min)/(u_max-u_min) - 1
        return new_u

    def normalize_c(self, c):
        new_c = 2*(c- self.env.c_bounds[:,0])/(self.env.c_bounds[:,1]-self.env.c_bounds[:,0]) - 1
        return new_c


    def normalize_y(self,Y):
        new_y = 2*(Y- self.env.y_bounds[:,0])/(self.env.y_bounds[:,1]-self.env.y_bounds[:,0]) - 1
        return new_y

    def normalize_state(self,state):
        min_max_state = np.concatenate([self.env.y_bounds,self.env.y_bounds,self.env.u_bounds,self.env.c_bounds],axis=0)
        state_min = min_max_state[:,0]
        state_max = min_max_state[:,1]
        new_state = 2*(state - state_min)/(state_max-state_min)-1
        return new_state




    def draw_uk_Jk(self, y, y_star, c, nc, title, act_list):
        return

        if self.step%5 !=0:
            return
        action = self.actor_nn(torch.cat([y, y_star, c], dim=1))

        # J(k+1) = U(k)+J(y(k+1),c)
        X, Y = np.meshgrid(
            np.linspace(-1,1,30),
            np.linspace(-1,1,30)
        )
        J_pred = self.critic_nn(torch.cat((y, y_star, nc, action), dim=1))
        cal_uk_Jk = lambda u1,u2:float(self.critic_nn(torch.cat((y, y_star, nc, torch.FloatTensor([[u1,u2]])
                                                                 ), dim=1)))
        f = np.frompyfunc(cal_uk_Jk, 2, 1)
        J_pred_res = f(X, Y)
        fig = plt.figure()
        plt.contourf(X, Y, J_pred_res, 16, alpha=.75, cmap='jet')
        plt.colorbar()
        act_array = np.array(act_list)
        plt.scatter(act_array[:,0], act_array[:, 1])
        ### 画每step一次，actor网络新输出的action
        for i in range(len(act_list)-1):
            self.drawArrow1(act_list[i], act_list[i+1],fig,i)
        #plt.scatter(act_list[:, 0],act_list[:, 1], marker='o', s=30, c='y')

        # C = contour(X, Y, J_pred_list, 8, colors='black', linewidth=.5)
        plt.title(title)
        plt.show()
        # print(mse_array)

    def drawArrow1(self, A, B, fig,iter):
        '''
        Draws arrow on specified axis from (x, y) to (x + dx, y + dy).
        Uses FancyArrow patch to construct the arrow.

        The resulting arrow is affected by the axes aspect ratio and limits.
        This may produce an arrow whose head is not square with its stem.
        To create an arrow whose head is square with its stem, use annotate() for example:
        Example:
            ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->"))
        '''
        ax = plt.gca()
        # fc: filling color
        # ec: edge color

        try:

            if np.linalg.norm((A, B))<0.000001:
                return
            ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
                     length_includes_head=True,  # 增加的长度包含箭头部分
                     head_width=0.05/(1+math.log(iter*0.2+1)), head_length=0.1/(1+math.log(iter+1)), fc='orange', )
        except Exception as e:
            print(e)

if __name__ == '__main__':

    env = Flotation(normalize=False)
    controller = ILPL(env=env,
                      u_bounds=env.u_bounds,
                      Vm=np.diag([0.1,1,1,0.1]),
                      Lm=np.diag([0.1,1])
                          )

