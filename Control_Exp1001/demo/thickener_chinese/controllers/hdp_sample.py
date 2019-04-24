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


class HDP_sample(ACBase):
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
                 hidden_critic = 10,
                 hidden_actor = 10,
                 predict_epoch = 35

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
        super(HDP_sample, self).__init__(gpu_id=gpu_id,replay_buffer=replay_buffer,
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


        dim_c = env.size_yudc[3]
        dim_y = env.size_yudc[0]
        dim_u = env.size_yudc[1]
        # Train model neural network
        self.model_nn = nn.Sequential(
            nn.Linear(dim_y+dim_u+dim_c, hidden_model),
            nn.Tanh(),
            nn.Linear(hidden_model, dim_y)
        )
        self.model_nn_optim = torch.optim.Adam(self.model_nn.parameters(), lr=model_nn_lr)
        #self.train_identification_model()

        #mse = self.test_predict_model(test_rounds=400)

        #定义actor网络相关
        self.actor_nn = nn.Sequential(
            nn.Linear(2*dim_y+dim_c, hidden_actor, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_actor, dim_u),
            nn.Tanh(),
           # nn.Linear(dim_u, dim_u)
        )

        self.actor_nn_optim = torch.optim.Adam(self.actor_nn.parameters(), lr=actor_nn_lr)



        #定义critic网络相关:HDP

        self.critic_nn = nn.Sequential(
            nn.Linear(dim_y+dim_y+dim_c, hidden_critic, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_critic, 1),
        )
        self.critic_nn_optim = torch.optim.Adam(self.critic_nn.parameters(), lr=critic_nn_lr)
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
        self.predict_epoch = predict_epoch



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

        if (self.step-1) % 30 ==0:
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

            # 定义TD loss
            critic_loss = self.critic_criterion(q_value, Variable(target_q.data))

            if critic_loss < self.critic_nn_error_limit:
                break

            #target_q.register_hook(lambda grad:print(grad))
            self.critic_nn_optim.zero_grad()
            critic_loss.backward()
            loop_time += 1
            self.critic_nn_optim.step()
            if loop_time >= 10000:
                break

            # endregion


        print('step:',self.step, 'critic loop',loop_time)
        loop_time = 0
        while True:
            # region update actor nn
            # y(k+1) = f(y(k),u(k),c(k))
            action = self.actor_nn(torch.cat([y,y_star,c],dim=1))
            y_pred = self.model_nn(torch.cat((y, action, c), dim=1))
            # J(k+1) = U(k)+J(y(k+1),c)

            S = torch.FloatTensor(self.env.penalty_calculator.S)
            U = torch.FloatTensor(self.mid_u)
            diff_U = torch.FloatTensor(self.u_bounds[:,1]-self.u_bounds[:,0])
            det_u = torch.nn.functional.linear(input=action, weight=torch.diag(diff_U/2))
            penalty_u = (det_u.mm(torch.FloatTensor(self.env.penalty_calculator.S)).mm(
                det_u.t()
            )).diag().unsqueeze(dim=1)
            J_pred = self.critic_nn(torch.cat((y_pred, y_star, nc), dim=1))
            J_loss = penalty_u + self.gamma * J_pred.mean()
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
            if abs(J_loss-last_J) < self.actor_nn_error_limit:
                break
            last_J = float(J_loss)
            # endregion

        print('step:',self.step, 'actor loop',loop_time)

    def u_grad_cal(self, grad):
        global u_grad
        u_grad = grad
    def y_grad_cal(self, grad):
        global y_pred_grad
        y_pred_grad = grad

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
            print(_)
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

        for i in range(self.env.size_yudc[0]):
            plt.plot(np.arange(real_y_array.shape[0]), real_y_array[:,i], 'o-')

            plt.plot(np.arange(real_y_array.shape[0]),pred_y_array[:,i],'r.--', linewidth=0.7)
            plt.legend(['Real Value','Forecast curve'])
            plt.title(self.env.y_name[i])
            plt.xlabel('')
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

    def data_generator(self, train_x, train_y):
        n = train_x.shape[0]
        tmp_x = np.copy((train_x)).squeeze()[:,4:6]
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

    def test_c_f(self,y1,y2):
        y = self.normalize_y(np.array([y1, y2]))
        c = self.normalize_c(np.array([40, 73]))
        y_star = self.normalize_y(np.array([1.48, 680]))
        input_critic = np.hstack([y, y_star, c])
        tmp = torch.FloatTensor(input_critic).unsqueeze(0)
        J_pred = self.critic_nn(tmp).squeeze().data
        return float(J_pred)

    def test_critic_nn(self,title):
        fig = plt.figure()
        y_min = self.y_min
        y_max = self.y_max

        X , Y = np.meshgrid(np.linspace(y_min[0], y_max[0], 30),
                            np.linspace(y_min[1], y_max[1], 30))
        f = np.frompyfunc(self.test_c_f,2,1)
        J_pred_res = f(X,Y)
        X1, Y1 = np.meshgrid(
            np.linspace(-1, 1, 30),
            np.linspace(-1, 1, 30)
        )
        plt.contourf(X1, Y1, J_pred_res, 16, alpha=.75, cmap='jet')
        plt.colorbar()
        #C = contour(X, Y, J_pred_list, 8, colors='black', linewidth=.5)
        plt.title((title))
        plt.show()






if __name__ == '__main__':

    env = Flotation(normalize=False)
    controller = ILPL(env=env,
                      u_bounds=env.u_bounds,
                      Vm=np.diag([0.1,1,1,0.1]),
                      Lm=np.diag([0.1,1])
                      )

