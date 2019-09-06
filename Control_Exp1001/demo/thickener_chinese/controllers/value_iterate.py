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
from Control_Exp1001.demo.thickener.ILPL.predict import Model
sys.path.append(('./'))
import itertools
from Control_Exp1001.demo.flotation.plotuilt import PltUtil
import  mpl_toolkits.mplot3d as p3d
from pylab import contourf
from pylab import contour

from scipy.optimize import minimize

class VI(ACBase):
    def __init__(self,

                 gpu_id=1,
                 replay_buffer = None,
                 u_bounds = None,
                 exploration = None,
                 env=None,
                 predict_training_rounds=10000,
                 gamma=0.6,

                 batch_size = 1,
                 predict_batch_size=32,

                 model_nn_error_limit = 0.08,
                 critic_nn_error_limit = 1,
                 actor_nn_error_limit = 0.1,

                 actor_nn_lr = 0.01,
                 critic_nn_lr = 0.01,
                 model_nn_lr = 0.01,

                 indice_y = None,
                 indice_u = None,
                 indice_y_star = None,
                 indice_c=None,
                 hidden_model = 10,
                 hidden_critic = 14,
                 hidden_actor = 10,
                 predict_epoch = 35,
                 Nc=500,
                 u_optim='sgd',
                 img_path="None",
                 test_period=100,

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
        super(VI, self).__init__(gpu_id=gpu_id,replay_buffer=replay_buffer,
                                   u_bounds=u_bounds,exploration=exploration)
        if env is None:
            env = Thickener()

        self.env=env
        self.predict_training_rounds = predict_training_rounds

        self.device = None
        self.cuda_device(gpu_id)
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size



        self.predict_training_losses = []
        self.model_nn = None
        self.model_nn_error_limit = model_nn_error_limit
        self.critic_nn_error_limit = critic_nn_error_limit
        self.actor_nn_error_limit = actor_nn_error_limit
        self.u_grad = [0, 0]
        self.y_grad = [0, 0]

        dim_c = env.size_yudc[3]
        dim_y = env.size_yudc[0]
        dim_u = env.size_yudc[1]

        # 定义critic网络相关:HDP

        self.critic_nn = nn.Sequential(
            nn.Linear(dim_y+dim_y+dim_c, hidden_critic, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_critic, 1),
        )

        self.critic_nn_optim = torch.optim.Adam(self.critic_nn.parameters(), lr=critic_nn_lr)
        self.critic_criterion = torch.nn.MSELoss()

        # Train model neural network
        self.model_nn = nn.Sequential(
            nn.Linear(dim_y+dim_u+dim_c, hidden_model),
            nn.Tanh(),
            nn.Linear(hidden_model, dim_y)
        )
        self.model_nn_optim = torch.optim.Adam(self.model_nn.parameters(), lr=model_nn_lr)
        #self.train_identification_model()

        #mse = self.test_predict_model(test_rounds=400)

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
        self.predict_epoch = predict_epoch
        self.S = torch.FloatTensor(self.env.penalty_calculator.S)
        self.u_grad = torch.zeros((1, 2))
        self.y_grad = torch.zeros((1, 2))
        self.u_optim = u_optim
        self.u_iter_times = 0
        self.log_y = []
        self.img_path = img_path
        self.Nc=Nc
        self.test_period = test_period




    def cuda_device(self, cuda_id):
        use_cuda = torch.cuda.is_available()
        cuda = 'cuda:'+str(cuda_id)
        self.device = torch.device(cuda if use_cuda else "cpu")

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

        act = torch.nn.Parameter(2*torch.rand((1,self.env.size_yudc[1]))-1)
        if self.u_optim is "adam":
            opt = torch.optim.Adam(params=[act], lr=0.1)
        elif self.u_optim is 'sgd':
            opt = torch.optim.SGD(params=[act], lr=0.4)
        elif self.u_optim is 'RMSprop':
            opt = torch.optim.RMSprop(params=[act], lr=0.01)
        elif self.u_optim is 'adagrad':
            opt = torch.optim.Adagrad(params=[act], lr=0.1)

        act_list = []
        act_list.append(np.copy(act.data.numpy()).squeeze())
        self.u_iter_times=0
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
            if torch.dist(act, old_act)<1e-4:
                break
            if self.u_iter_times>8000:
                break

        #print('step:',self.step, 'find u loop', self.u_iter_times)

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

        if self.step % self.test_period == 0 :
            self.test_critic_nn(title='round:'+str(self.step), cur_state=y, act_list=act_list)
        # if 195<self.step-1<220:
        #     self.test_critic_nn(title='round:'+str(self.step), cur_state=y, act_list=act_list)

        return act



    def _train(self, s, u, ns, r, done):

        #print(r)
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
        self.log_y.append(y.clone().numpy()[-1,:])
        self.y_grad_arrow = None



        # self.test_critic_nn(title='round:'+str(self.step), cur_state=y[-1], act_list=None)
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


        # 循环更新critic网路
        loop_time = 0
        last_J = np.inf
        # region update critic nn
        while True:
            y.requires_grad = True
            q_value = self.critic_nn(torch.cat((y, y_star, c), dim=1))


            next_q_value = self.critic_nn(torch.cat((ny, y_star, nc), dim=1))
            target_q = penalty + self.gamma * next_q_value
            #print(target_q)


            loop_time += 1
            # 定义TD loss
            critic_loss = self.critic_criterion(q_value, Variable(target_q.data))
            #critic_loss = self.critic_criterion(q_value, target_q)


            target_q.register_hook(lambda grad:print(grad))
            self.critic_nn_optim.zero_grad()
            critic_loss.backward()
            self.critic_nn_optim.step()
            self.y_grad_arrow = np.copy(y.grad.data.numpy()[-1])
            y.grad.data = torch.zeros(y.grad.data.shape)
            if loop_time >= self.Nc:
                break

            if critic_loss < self.critic_nn_error_limit:
                break
            # endregion
        #print('step:',self.step, 'critic loop',loop_time)

        #self.test_critic_nn(title='round:'+str(self.step), cur_state=y[-1], act_list=None)



    def u_grad_cal(self, grad):
        global u_grad
        self.u_grad = grad
    def y_grad_cal(self, grad):
        global y_pred_grad
        y_pred_grad = grad
        self.y_grad = grad

    def predict(self, state, act):
        cur_y=state[2:4]
        c = state[self.indice_c]
        x = torch.FloatTensor(self.normalize_x(np.hstack([cur_y, act, c])))

        y = self.model_nn.forward(x)
        delta_y = torch.FloatTensor(self.y_max-self.y_min)
        min_y = torch.FloatTensor(self.y_min)
        y = (y + 1)*(delta_y)/2+min_y
        return y.squeeze(dim=0)

    def cal_training_data(self, rounds=None, new=False):
        """

        :return:
        """
        if rounds is None:
            rounds = self.predict_training_rounds

        # 写在json里暂存，防止每次都靠仿真模型太慢
        json_path = "training_data_" + str(rounds) + '.json'
        if os.path.exists(json_path) and not new:
            with open(json_path, 'r',) as fp:
                train_x, train_y = json.load(fp)
                train_x = np.array(train_x)
                train_y = np.array(train_y)
                return train_x, train_y

        train_x = []
        train_y = []
        # 生成训练数据
        print("模拟生成")
        for _ in range(rounds):
            #print(_)
            y = self.env.observation()[2:4]
            act = np.random.uniform(self.u_bounds[:,0], self.u_bounds[:,1])
            c = self.env.observation()[6:8]

            train_x.append(np.hstack([y, act, c])[np.newaxis,:])
            self.env.step(act)
            new_state = self.env.observation()[2:4]
            train_y.append(new_state[np.newaxis, :])
            if random.random() < 0.0005:
                self.env.reset()
        # 写json暂存
        with open(json_path, 'w',) as fp:
            tmp_x = np.copy(train_x).tolist()
            tmp_y = np.copy(train_y).tolist()
            json.dump((tmp_x, tmp_y), fp)

        return train_x, train_y

    def test_predict_model(self, test_rounds=300,diff=False):
        """
        测试预测模型效果的，画出差分图
        :param test_rounds:
        :return:
        """
        self.env.reset()
        pred_y_list = []
        real_y_list = []
        #pred_y_list.append(self.env.observation()[2:4][np.newaxis,:])
        for _ in range(test_rounds):
            act = np.random.uniform(self.u_bounds[:,0],
                                    self.u_bounds[:,1])
            pred_y = self.predict(self.env.observation(), act)

            old_y = self.env.observation()[2:4]
            if not diff:
                old_y = old_y*0
            pred_y_list.append(pred_y.detach().numpy() - old_y)
            self.env.step(act)
            real_y_list.append(self.env.observation()[2:4] - old_y)
            if random.random() < 0.0005:
                self.env.reset()
        real_y_array = np.array(real_y_list)
        pred_y_array = np.array(pred_y_list)
        self.con_predict_mse = mean_squared_error(real_y_array[:,1], pred_y_array[:,1])

        for i in range(self.env.size_yudc[0]):
            plt.plot(np.arange(real_y_array.shape[0]), real_y_array[:,i], 'o-')

            plt.plot(np.arange(real_y_array.shape[0]),pred_y_array[:,i],'r.--', linewidth=0.7)
            plt.legend(['Real Curve','Forecast Curve'])
            plt.title(self.env.y_name[i])
            plt.xlabel('')

            plt.savefig(os.path.join('../images/',self.img_path)+'/'+str(self.env.y_name[i])+"_predict"+
                        self.__class__.__name__+'.png', dpi=300)

            plt.show()

    def cal_predict_mse(self, test_rounds=1000, diff=False):

        self.env.reset()
        #X, Y = self.cal_training_data(test_rounds, new=True)
        X, Y = self.cal_training_data(test_rounds, new=False)
        X, Y = self.normalize(X, Y)
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        pred_y = self.model_nn(X)
        loss = self.model_criterion(pred_y, Y)

        return loss

    def define_identification_nn(self, Vm, Lm):
        self.model_nn = Model(dim_in=self.env.size_yudc[0]+self.env.size_yudc[1]+self.env.size_yudc[3],
                              dim_out=self.env.size_yudc[0],dim_hidden=6,device=self.device,Vm=Vm,Lm=Lm)


    def normalize_u(self, u):
        new_u = 2*(u- self.x_min[2:4])/(self.x_max[2:4]-self.x_min[2:4]) - 1
        return new_u

    def normalize_c(self, c):
        new_c = 2*(c- self.x_min[4:6])/(self.x_max[4:6]-self.x_min[4:6]) - 1
        return new_c

    def normalize_x(self,X):
        new_x = 2*(X- self.x_min)/(self.x_max-self.x_min) - 1

        return new_x.reshape(-1)

    def normalize_y(self,Y):
        new_y = 2*(Y- self.y_min)/(self.y_max-self.y_min) - 1
        return new_y.reshape(-1)

    def normalize_state(self,state):
        state = np.array(state)
        state_max = np.hstack([self.y_max, self.x_max])
        state_min = np.hstack([self.y_min, self.x_min])
        new_state = 2*(state - state_min)/(state_max-state_min) - 1
        return new_state

    def normalize(self, train_x, train_y, new_para=False):
        if new_para:
            self.x_max = np.squeeze(np.max(train_x, 0))
            self.x_min = np.squeeze(np.min(train_x, 0))
            self.y_max = np.squeeze(np.max(train_y, 0))
            self.y_min = np.squeeze(np.min(train_y, 0))
        new_x = 2*(train_x- self.x_min)/(self.x_max-self.x_min) - 1
        new_y = 2*(train_y- self.y_min)/(self.y_max-self.y_min) - 1
        return new_x, new_y

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
        torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))

        # 建立数据集
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.predict_batch_size,shuffle=True)

        mse_list = []
        # 训练预测模型
        for epoch in itertools.count():

            #print("Epoch:{}".format(epoch+1), "")
            sum_loss = 0
            for step,(batch_x, batch_y) in enumerate(loader):
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                output = self.model_nn(b_x)
                loss = self.model_criterion(output, b_y)
                self.model_nn_optim.zero_grad()
                loss.backward()

                self.model_nn_optim.step()
                self.predict_training_losses.append(loss.item())
                sum_loss += loss.item()


            #print("Loss:{}".format(sum_loss))
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

    def test_c_f(self,y1,y2):
        y = np.array([y1, y2])

        if self.step<200:
            c = self.normalize_c(np.array([40, 73]))
        else:
            c = self.normalize_c(np.array([35, 65]))
        y_star = self.normalize_y(np.array([1.48, 680]))
        input_critic = np.hstack([y, y_star, c])
        tmp = torch.FloatTensor(input_critic).unsqueeze(0)
        J_pred = self.critic_nn(tmp).squeeze().data
        return float(J_pred)

    def test_c_f_u(self,u1,u2):

        act = torch.FloatTensor([[u1, u2]])
        diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
        det_u = torch.nn.functional.linear(input=act, weight=torch.diag(diff_U/2))
        # penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
        #     det_u.t()
        # )).diag().unsqueeze(dim=1)
        penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
            det_u.t()
        )).diag()
        # if self.step<200:
        #     c = self.normalize_c(np.array([40, 73]))
        # else:
        #     c = self.normalize_c(np.array([35, 65]))
        c = self.normalize_c(self.env.c)
        c=torch.FloatTensor([c])
        y = self.sstate
        y= y.squeeze().unsqueeze(0)
        y_star = self.normalize_y([1.48, 680])
        y_star = torch.FloatTensor([y_star])
        y_pred = self.model_nn(torch.cat((y, act, c), dim=1))
        J_pred = self.critic_nn(torch.cat((y_pred, y_star, c), dim=1))
        #penalty_u = torch.zeros(J_pred.shape)
        J_loss = penalty_u + self.gamma * J_pred
        #J_loss = J_pred
        return float(J_loss)

    def test_critic_nn(self,cur_state=None, title="None",act_list=None):

        # if  self.step-1<200 :
        #     return
        if not self.step %self.test_period == 0:
            return

        act_list = np.array(act_list)
        if act_list is None:
            act_list=[]
        fig = plt.figure()
        y_min = self.y_min
        y_max = self.y_max

        X , Y = np.meshgrid(np.linspace(y_min[0], y_max[0], 30),
                            np.linspace(y_min[1], y_max[1], 30))

        X1, Y1 = np.meshgrid(
            np.linspace(-1, 1, 60),
            np.linspace(-1, 1, 60)
        )
        f = np.frompyfunc(self.test_c_f,2,1)
        self.sstate = cur_state
        f_u = np.frompyfunc(self.test_c_f_u,2,1)
        J_pred_res = f(X1,Y1)
        J_pred_res_u = f_u(X1, Y1)
        fig = plt.figure()
        plt.contourf(X1, Y1, J_pred_res, 40, alpha=.75, cmap='jet')

        plt.colorbar()
        #C = contour(X, Y, J_pred_list, 8, colors='black', linewidth=.5)
        plt.title((title))



        cor_list = ['k','g']
        if len(self.replay_buffer)>1:
            cor_list = ['orange','r']

        tmp_log_y = np.array(self.log_y)
        for i in range(min(len(tmp_log_y),2)):

            plt.scatter(tmp_log_y[-1-i, 0], tmp_log_y[-1-i,1], s=40,
                        c=cor_list[i],label='[h(k-'+str(i+1)+')' + ',y(k-'+str(i+1)+')]')
        plt.legend()
        root_path = os.path.join('../images/', self.img_path)
        if not os.path.exists(os.path.join(root_path, 'iters_trajectory/sum/')):
            os.mkdir(os.path.join(root_path, 'iters_trajectory/'))
            os.mkdir(os.path.join(root_path, 'iters_trajectory/sum/'))
        plt.savefig(os.path.join(root_path, 'iters_trajectory/sum/')+str(self.batch_size)+'-'+str(self.step)+'.eps', format='eps',dpi=600)
        plt.show()
        ### 画一个y(k)梯度方向的箭头
        # 效果太差了，难以掌握梯度大小，很多梯度方向不准确
        # if not self.y_grad_arrow is None:
        #     begin_point =tmp_log_y[-1]
        #     arrow_norm = np.linalg.norm(self.y_grad_arrow)
        #     new_norm = math.log(arrow_norm + 1)
        #     ratio = new_norm/arrow_norm
        #     end_point =tmp_log_y[-1]+self.y_grad_arrow * ratio
        #     plt.annotate("", xy=(end_point[0], end_point[1]),
        #                  xytext=(begin_point[0], begin_point[1]),
        #                  arrowprops=dict(arrowstyle="->"))
        #     self.y_grad_arrow = None
        ########################


        # 画迭代计算u的过程
        # 绘制横纵坐标为U的图
        if act_list is None:
            return
        plt.contourf(X1, Y1, J_pred_res_u, 32, alpha=.75, cmap='jet')
        plt.colorbar()
        #C = contour(X, Y, J_pred_list, 8, colors='black', linewidth=.5)
        plt.title(title)
        for i in range(len(act_list)-1):
            self.draw_arrow(act_list[i], act_list[i+1],fig,i)
        #plt.scatter(act_list[:, 0],act_list[:, 1], marker='o', s=30, c='y')

        root_path = os.path.join('../images/', self.img_path,'u_distribution')
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        plt.savefig(os.path.join(root_path, str(self.batch_size)+'-'+str(self.step)+'.eps'), format='eps',dpi=600)
        plt.show()
        #plt.show()

    def draw_arrow(self, A, B, fig,iter):
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



        # 注意： 默认显示范围[0,1][0,1],需要单独设置图形范围，以便显示箭头







if __name__ == '__main__':

    env = Flotation(normalize=False)
    controller = ILPL(env=env,
                      u_bounds=env.u_bounds,
                      Vm=np.diag([0.1,1,1,0.1]),
                      Lm=np.diag([0.1,1])
                      )

