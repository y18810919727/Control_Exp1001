#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np

import torch
from matplotlib import pyplot as plt
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json
from torch import nn
from Control_Exp1001.demo.thickener.controllers.hdp import HDP


class DHP(HDP):
    def __init__(self,
                test_period=1,
                **para,
                ):
        super(DHP, self).__init__(**para)
        env = para['env']
        dim_c = env.size_yudc[3]
        dim_y = env.size_yudc[0]
        hidden_critic = para['hidden_critic']
        critic_nn_lr = para['critic_nn_lr']
        self.test_period=test_period
        self.critic_nn = nn.Sequential(
            nn.Linear(dim_y+dim_y+dim_c, hidden_critic, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_critic, dim_y,bias=False),
        )
        self.critic_nn_optim = torch.optim.SGD(self.critic_nn.parameters(), lr=critic_nn_lr)
        self.critic_criterion = torch.nn.MSELoss()



    # 计算x*A.x^T，x为矩阵，每一行代表一个状态
    def cal_x_penalty(self, x):

        y_star = torch.FloatTensor(self.env.y_star)

        weight_matrix = torch.FloatTensor(self.env.penalty_calculator.weight_matrix)
        det_x = torch.FloatTensor(self.y_max - self.y_min)
        mid_x = torch.FloatTensor(self.y_max + self.y_min)/2
        con_normal_x = (x/2).mm(det_x.diag()) + mid_x  # 反归一化
        bias_x = con_normal_x - y_star  # 反归一化的x与目标值的偏差
        penalty_x = bias_x.mm(weight_matrix).mm(bias_x.t()).diag()
        return penalty_x


    # 计算u*B.u^T，u为矩阵，每一行代表一个状态
    def cal_u_penalty(self, action):

        diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
        det_u = torch.nn.functional.linear(input=action, weight=torch.diag(diff_U/2))
        penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
            det_u.t()
        )).diag().unsqueeze(dim=1)
        return penalty_u




    def update_model(self,state, action, penalty, next_state, done):

        if self.step >80:
            print('11')
            pass

        state = torch.FloatTensor(self.normalize_state(state)).to(self.device)
        next_state = torch.FloatTensor(self.normalize_state(next_state)).to(self.device)
        penalty = torch.FloatTensor(penalty).unsqueeze(1).to(self.device)
        indices_y = torch.LongTensor(self.indice_y)
        indices_c = torch.LongTensor(self.indice_c)
        indices_y_star = torch.LongTensor(self.indice_y_star)
        y = torch.index_select(state, 1, indices_y)
        y_star = torch.index_select(state, 1, indices_y_star)

        c = torch.index_select(state, 1, indices_c)
        nc = torch.index_select(next_state, 1, indices_c)

        # if (self.step-1) % 30 == 0:
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

        self.update_critic(y, y_star, c, nc)
        self.update_actor(y ,y_star, c, nc)


    def update_critic(self, y, y_star, c, nc):

        # 更新critic网路
        loop_time = 0
        #last_J = np.inf
        # region update critic nn
        while True:

            # J(k+1) = U(k)+J(y(k+1),c)



            xk = y.clone()
            xk.requires_grad = True

            action = self.actor_nn(torch.cat([xk, y_star,c],dim=1)) # create action by uk
            #####################################
            # region 生成控制量U(k)惩罚的计算图


            # 计算U(k)中u的起作用的部分
            penalty_u = self.cal_u_penalty(action)
            # 计算U(k)中x起作用的部分
            penalty_x = self.cal_x_penalty(x=xk)

            penalty_all = penalty_u + penalty_x
            # endregion 结束控制量惩罚图
            #####################################
            penalty_all_loss = penalty_all.sum()
            penalty_all_loss = penalty_all_loss / self.batch_size
            penalty_all_loss.backward(retain_graph=True)

            #####################################
            # region 生成控制量J(k+1)的计算图
            x_pred = self.model_nn(torch.cat((xk, action, c), dim=1))
            x_pred_costate = torch.tensor(self.critic_nn(torch.cat((x_pred, y_star, nc), dim=1)))
            # endregion 结束生成J(k+1)计算图
            #####################################

            x_pred.backward(self.gamma*x_pred_costate/self.batch_size) #求J(k+1)对xk的梯度


            # region 更新critic 网络
            x_costate = self.critic_nn(torch.cat((y, y_star, c), dim=1))
            x_costate_real = torch.FloatTensor(xk.grad)
            critic_loss = self.critic_criterion(x_costate, x_costate_real)
            self.critic_nn_optim.zero_grad()
            critic_loss.backward()
            self.critic_nn_optim.step()

            # endregion

            loop_time += 1
            # 定义TD loss
            self.critic_nn_optim.step()
            if loop_time >= self.Nc:
                break

            if critic_loss < self.critic_nn_error_limit:
                break
            # endregion
        print('step:',self.step, 'critic loop',loop_time)
        self.test_critic_nn(title='round-'+str(self.step))

    def update_actor(self, y, y_star, c, nc):

        loop_time = 0
        while True:
            # region update actor nn
            # y(k+1) = f(y(k),u(k),c(k))
            action = self.actor_nn(torch.cat([y,y_star,c],dim=1))
            y_pred = self.model_nn(torch.cat((y, action, c), dim=1))
            # J(k+1) = U(k)+J(y(k+1),c)

            y_pred.register_hook(self.y_grad_cal)
            action.register_hook(self.u_grad_cal)


            # 计算u项的惩罚
            penalty_u = self.cal_u_penalty(action)

            y_pred_costate = self.critic_nn(torch.cat((y_pred, y_star, nc), dim=1))

            penalty_u = penalty_u.sum()/self.batch_size
            penalty_u.backward(retain_graph=True)
            self.U2u_grad = self.u_grad.clone()
            J_loss = self.gamma * (y_pred.mul(y_pred_costate).sum())
            J_loss = J_loss / self.batch_size
            #J_loss = self.gamma * (y_pred.mul(y_pred_costate).mean()*2)*10
            #action.requires_grad = True
            self.actor_nn_optim.zero_grad()
            J_loss.backward()

            self.J2u_grad = self.u_grad-self.U2u_grad
            self.J2y_grad = self.y_grad

            self.actor_nn_optim.step()

            #print('critic loss', critic_loss)
            loop_time += 1
            # if abs(J_loss-last_J) < self.actor_nn_error_limit:
            #     break
            last_J = float(J_loss)
            # if J_loss < 1e-4:
            #     break
            if loop_time > self.Na:
                break
            # endregion

        print('step:',self.step, 'actor loop',loop_time)

    def critic_out_test(self, y1, y2):
        y = np.array([y1, y2])
        c = self.normalize_c(self.env.c)
        y_star = self.normalize_y(self.env.y_star)
        input_critic = np.hstack([y, y_star, c])
        tmp = torch.FloatTensor(input_critic).unsqueeze(0)
        cos_state_pred = self.critic_nn(tmp).squeeze().data

        return cos_state_pred[0], cos_state_pred[1]

    def plt_contourf(self, title, X, Y, costate):

        plt.contourf(X,Y,costate, 40, alpha=.75, cmap='jet')
        plt.title(title)
        plt.colorbar()

    def plt_state_trajectory(self, log_y, order):

        cor_list = ['k','g']
        if len(self.replay_buffer)>1:
            cor_list = ['orange','r']

        tmp_log_y = np.array(log_y)
        for i in range(min(len(tmp_log_y),2)):

            plt.scatter(tmp_log_y[-1-i, 0], tmp_log_y[-1-i,1], s=40,
                        c=cor_list[i],label='[h(k-'+str(i+1)+')' + ',y(k-'+str(i+1)+')]')
        plt.legend()
        plt.text(x=0,y=0,s=str(self.critic_out_test(tmp_log_y[-1, 0], tmp_log_y[-1,1])[order]), fontsize=30)

    def test_critic_nn(self,title):
        if not self.step %self.test_period == 0:
            return
        X,Y = np.meshgrid(
            np.linspace(-1,1,30), np.linspace(-1,1,30)
                    )
        f_c = np.frompyfunc(self.critic_out_test, 2, 2)
        costate = f_c(X,Y)

        fig = plt.figure(1, (18,6))
        fig.add_subplot(121)
        self.plt_contourf(title=title+'costate of Height',X=X, Y=Y, costate=costate[0])
        self.plt_state_trajectory(self.log_y, order=0)

        fig.add_subplot(122)
        self.plt_contourf(title=title+'costate of Concentration',X=X, Y=Y, costate=costate[1])
        self.plt_state_trajectory(self.log_y, order=1)
        plt.show()
        return



