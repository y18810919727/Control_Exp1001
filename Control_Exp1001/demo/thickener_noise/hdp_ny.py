#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json
from Control_Exp1001.demo.thickener.hdp import HDP
import sys
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
sys.path.append(('./'))



class HdpNy(HDP):

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

        if (self.step-1) % 30 == 0:
            self.test_critic_nn(title='round:'+str(self.step))


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
        while True:
            q_value = self.critic_nn(torch.cat((y, y_star, c), dim=1))


            next_q_value = self.critic_nn(torch.cat((ny, y_star, nc), dim=1))
            target_q = penalty + self.gamma * next_q_value
            #print(target_q)


            loop_time += 1
            # 定义TD loss
            critic_loss = self.critic_criterion(q_value, Variable(target_q.data))


            target_q.register_hook(lambda grad:print(grad))
            self.critic_nn_optim.zero_grad()
            critic_loss.backward()
            self.critic_nn_optim.step()
            if loop_time >= 1000:
                break

            if critic_loss < self.critic_nn_error_limit:
                break
            # endregion
        print('step:',self.step, 'critic loop',loop_time)


        loop_time = 0
        while True:
            # region update actor nn
            # y(k+1) = f(y(k),u(k),c(k))
            action = self.actor_nn(torch.cat([ny,y_star,c],dim=1))
            y_pred = self.model_nn(torch.cat((ny, action, c), dim=1))
            # J(k+1) = U(k)+J(y(k+1),c)

            # 计算控制量惩罚的计算图
            S = torch.FloatTensor(self.env.penalty_calculator.S)
            U = torch.FloatTensor(self.mid_u)
            diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
            det_u = torch.nn.functional.linear(input=action, weight=torch.diag(diff_U/2))
            penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
                det_u.t()
            )).diag().unsqueeze(dim=1)

            J_pred = self.critic_nn(torch.cat((y_pred, y_star, nc), dim=1))
            J_loss = penalty_u + self.gamma * J_pred
            J_loss = J_loss.mean()
            self.actor_nn_optim.zero_grad()
            #action.requires_grad = True
            global u_grad
            global y_pred_grad
            y_pred.register_hook(self.y_grad_cal)
            action.register_hook(self.u_grad_cal)
            J_loss.backward()
            self.actor_nn_optim.step()

            #print('critic loss', critic_loss)
            loop_time += 1
            # if abs(J_loss-last_J) < self.actor_nn_error_limit:
            #     break
            last_J = float(J_loss)
            if J_loss < 1e-4:
                break
            if loop_time > 100:
                break
            # endregion

        print('step:',self.step, 'actor loop',loop_time)
