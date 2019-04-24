#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
from matplotlib import pyplot as plt
import torch
import json
from Control_Exp1001.demo.thickener.controllers.dhp import DHP
from Control_Exp1001.demo.thickener.controllers.value_iterate import VI


class DhpVI(DHP, VI):
    def __init__(self,
                 max_u_iters=1000,
                 policy_visual_period=1,
                 **para):
        super(DhpVI, self).__init__(**para)
        self.max_u_iters = max_u_iters
        self.policy_visual_period = policy_visual_period

    def _act(self, state):

        y = self.normalize_y(state[self.indice_y])
        y_star = self.normalize_y(state[self.indice_y_star])
        c = self.normalize_c(state[self.indice_c])

        self.log_y.append(np.copy(y))
        y = torch.FloatTensor(y).unsqueeze(0)
        y_star = torch.FloatTensor(y_star).unsqueeze(0)
        c = torch.FloatTensor(c).unsqueeze(0)


        self.delta_u = self.env.u_bounds[:, 1] - self.env.u_bounds[:, 0]
        self.mid_u = np.mean(self.env.u_bounds, axis=1)
        x = torch.FloatTensor(np.hstack((y, y_star,c))).unsqueeze(0)

        act = torch.nn.Parameter(2*torch.rand((1,self.env.size_yudc[1]))-1)
        opt = torch.optim.SGD(params=[act], lr=0.03)

        act_list = []
        act_list.append(np.copy(act.data.numpy()).squeeze())
        iters_cnt = 0
        while True:
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
            J_costate = self.critic_nn(torch.cat((y_pred, y_star, c), dim=1))
            #penalty_u = torch.zeros(J_pred.shape)
            J_loss = penalty_u + self.gamma*torch.mul(y_pred, J_costate).sum()
            J_loss = J_loss.mean()
            opt.zero_grad()
            J_loss.backward()
            opt.step()
            act.data = torch.nn.Parameter(torch.clamp(act,min=-1,max=1)).data
            act_list.append(np.copy(act.data.numpy()).squeeze())
            if torch.dist(act, old_act)<1e-8:
                break
            iters_cnt += 1
            if iters_cnt >= self.max_u_iters:
                break

        print('step:',self.step, 'find u loop', iters_cnt)
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
        if self.step % self.policy_visual_period == 0:
            self.policy_visual(title='round '+str(self.step)+' find u', act_list=act_list)
        return act

    def critic_u(self,u1, u2):
        y = torch.FloatTensor(self.normalize_y(self.env.y)).unsqueeze(0)
        act = torch.FloatTensor([[u1, u2]])
        act.requires_grad = True
        diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
        det_u = torch.nn.functional.linear(input=act, weight=torch.diag(diff_U/2))
        # penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
        #     det_u.t()
        # )).diag().unsqueeze(dim=1)
        penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
            det_u.t()
        )).diag()
        c = self.normalize_c(self.env.c)
        y_star = self.normalize_y(self.env.y_star)
        c=torch.FloatTensor([c])
        y_star = torch.FloatTensor([y_star])
        y_pred = self.model_nn(torch.cat((y, act, c), dim=1))
        J_costate = self.critic_nn(torch.cat((y_pred, y_star, c), dim=1))
        #penalty_u = torch.zeros(J_pred.shape)
        J_loss = penalty_u + self.gamma*torch.mul(y_pred, J_costate).sum()
        J_loss.backward()
        u_grad = act.grad.data.squeeze(0)
        return float(u_grad[0]), float(u_grad[1])

    def plt_act_list(self, act_list, fig, order):

        for i in range(len(act_list)-1):
            self.draw_arrow(act_list[i], act_list[i+1],fig,i)

        plt.text(x=0,y=0,s=str(self.critic_u(act_list[-1][0], act_list[-1][1])[order]), fontsize=30)
        #plt.scatter(act_list[:, 0],act_list[:, 1], marker='o', s=30, c='y')

    def policy_visual(self, title, act_list):
        X,Y = np.meshgrid(
            np.linspace(-1,1,30),
            np.linspace(-1,1,30)
        )
        test_f = np.frompyfunc(self.critic_u, 2, 2)

        costate = test_f(X,Y)

        fig = plt.figure(1, (18,6))
        fig.add_subplot(121)
        self.plt_contourf(title=title+' costate of fu',X=X, Y=Y, costate=costate[0])
        self.plt_act_list(act_list, fig, order=0)

        fig.add_subplot(122)
        self.plt_contourf(title=title+' costate of ff',X=X, Y=Y, costate=costate[1])
        self.plt_act_list(act_list, fig, order=1)
        plt.show()

    # def draw_arrow1(self, A, B, fig,iter):
    #     '''
    #     Draws arrow on specified axis from (x, y) to (x + dx, y + dy).
    #     Uses FancyArrow patch to construct the arrow.
    #
    #     The resulting arrow is affected by the axes aspect ratio and limits.
    #     This may produce an arrow whose head is not square with its stem.
    #     To create an arrow whose head is square with its stem, use annotate() for example:
    #     Example:
    #         ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
    #         arrowprops=dict(arrowstyle="->"))
    #     '''
    #     ax = fig.gca()
    #     # fc: filling color
    #     # ec: edge color
    #
    #     try:
    #
    #         if np.linalg.norm((A, B))<0.000001:
    #             return
    #         ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
    #                  length_includes_head=True,  # 增加的长度包含箭头部分
    #                  head_width=0.05/(1+math.log(iter*0.2+1)), head_length=0.1/(1+math.log(iter+1)), fc='orange', )
    #     except Exception as e:
    #         print(e)



    def update_actor(self, y, y_star, c, nc):
        pass
