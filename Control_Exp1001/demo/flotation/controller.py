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
from torch.autograd import Variable
import torch
import torch.optim as optim
from Control_Exp1001.control.base_ac import ACBase
from Control_Exp1001.simulation.flotation import Flotation
from Control_Exp1001.demo.flotation.critic import Critic
from Control_Exp1001.demo.flotation.actor import Actor
from Control_Exp1001.demo.flotation.predict import Model
sys.path.append(('./'))
import itertools
from Control_Exp1001.demo.flotation.plotuilt import PltUtil



class ILPL(ACBase):
    def __init__(self,

                 gpu_id=1,
                 replay_buffer = None,
                 u_bounds = None,
                 exploration = None,
                 env=None,
                 predict_training_rounds=10000,
                 Vm=None,
                 Lm=None,
                 Va=None,
                 La=None,
                 Vc=None,
                 Lc=None,
                 gamma=0.9,

                 batch_size = 1,
                 predict_batch_size=32,
                 model_nn_error_limit = 0.08,
                 critic_nn_error_limit = 1,
                 actor_nn_loss = 0.1,

                 u_iter=30,
                 u_begin = None,
                 indice_y = None,
                 indice_y_star = None,
                 u_first = None


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
        super(ILPL, self).__init__(gpu_id=gpu_id,replay_buffer=replay_buffer,
                                   u_bounds=u_bounds,exploration=exploration)
        if env is None:
            env = Flotation()

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
        self.actor_nn_error_limit = actor_nn_loss

        self.u_iter = u_iter

        # Train model neural network
        self.train_identification_model(Vm=Vm,Lm=Lm)
        self.test_predict_model(test_rounds=400)

        #定义actor网络相关
        self.actor_nn = None
        self.actor_nn_init(Va=Va,La=La)


        #定义critic网络相关
        self.critic_nn = None
        self.critic_nn_init(Vc=Vc,Lc=Lc)

        self.gamma = gamma
        self.u_begin = u_begin

        if indice_y is None:
            indice_y = [2,3]
        if indice_y_star is None:
            indice_y_star = [0,1]
        self.indice_y = indice_y
        self.indice_y_star = indice_y_star

        if u_first is None:
            u_first = np.array([1.8, 19])
        self.u_first = u_first
        self.first_act = True


        # 用来画图用
        self.u0_plt = PltUtil()
        self.u1_plt = PltUtil()
        self.y0_plt = PltUtil()
        self.y1_plt = PltUtil()
        self.wa_plt = PltUtil()
        self.wm_plt = PltUtil()
        self.wc_plt = PltUtil()








    def cuda_device(self, cuda_id):
        use_cuda = torch.cuda.is_available()
        cuda = 'cuda:'+str(cuda_id)
        self.device = torch.device(cuda if use_cuda else "cpu")

    def _act(self, state):

        self.y0_plt.push("Lcg*", state[0])
        self.y1_plt.push("Ltg*", state[1])
        self.y0_plt.push("Lcg", state[2])
        self.y1_plt.push("Ltg", state[3])
        # 第一次控制不用actor模型输出，否则会很离谱
        if self.first_act:

            self.u0_plt.push("hp", self.u_first[0])
            self.u1_plt.push("qa", self.u_first[1])
            self.first_act = False
            act = self.u_first

        # 用actor网络计算输出
        else:
            y = state[self.indice_y]
            y_star = state[self.indice_y_star]

            x = torch.FloatTensor(np.hstack((y, y_star))).unsqueeze(0)
            act = self.actor_nn(x).detach().squeeze(0).numpy()


        self.u0_plt.push("hp", act[0])
        self.u0_plt.push("hp min", self.u_bounds[0,0])
        self.u0_plt.push("hp max", self.u_bounds[0,1])

        self.u1_plt.push("qa", act[1])
        self.u1_plt.push("qa min", self.u_bounds[1,0])
        self.u1_plt.push("qa max", self.u_bounds[1,1])

        return act



    def _train(self, s, u, ns, r, done):

        # 先放回放池
        self.replay_buffer.push(s, u, r, ns, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        # 从回放池取数据，默认1条
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # 更新模型
        self.update_model(state, action, reward, next_state, done)

    def add_nnw2plt(self):
        self.wm_plt.push("wm11", float(self.model_nn.Wm.weight.data[0,0]))
        self.wm_plt.push("wm12", float(self.model_nn.Wm.weight.data[0,1]))
        self.wm_plt.push("wm13", float(self.model_nn.Wm.weight.data[0,2]))
        self.wm_plt.push("wm14", float(self.model_nn.Wm.weight.data[0,3]))
        self.wm_plt.push("wm21", float(self.model_nn.Wm.weight.data[1,0]))
        self.wm_plt.push("wm22", float(self.model_nn.Wm.weight.data[1,1]))
        self.wm_plt.push("wm23", float(self.model_nn.Wm.weight.data[1,2]))
        self.wm_plt.push("wm24", float(self.model_nn.Wm.weight.data[1,3]))

        self.wa_plt.push("wa11", float(self.actor_nn.Wa.weight.data[0,0]))
        self.wa_plt.push("wa12", float(self.actor_nn.Wa.weight.data[0,1]))
        self.wa_plt.push("wa13", float(self.actor_nn.Wa.weight.data[0,2]))
        self.wa_plt.push("wa14", float(self.actor_nn.Wa.weight.data[0,3]))
        self.wa_plt.push("wa21", float(self.actor_nn.Wa.weight.data[1,0]))
        self.wa_plt.push("wa22", float(self.actor_nn.Wa.weight.data[1,1]))
        self.wa_plt.push("wa23", float(self.actor_nn.Wa.weight.data[1,2]))
        self.wa_plt.push("wa24", float(self.actor_nn.Wa.weight.data[1,3]))

        self.wc_plt.push("wc1", float(self.critic_nn.Wc.weight.data[0,0]))
        self.wc_plt.push("wc2", float(self.critic_nn.Wc.weight.data[0,1]))
        self.wc_plt.push("wc3", float(self.critic_nn.Wc.weight.data[0,2]))
        self.wc_plt.push("wc4", float(self.critic_nn.Wc.weight.data[0,3]))


    def update_model(self,state, action, penalty, next_state, done):

        self.add_nnw2plt()

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        penalty = torch.FloatTensor(penalty).unsqueeze(1).to(self.device)
        indices_y = torch.LongTensor(self.indice_y)
        indices_y_star = torch.LongTensor(self.indice_y_star)
        y = torch.index_select(state, 1, indices_y)
        ny = torch.index_select(next_state, 1, indices_y)
        y_star = torch.index_select(state, 1, indices_y_star)


        # region update model nn
        while True:

            next_state_predict = self.model_nn(torch.cat((y, action), dim=1))
            model_loss = self.model_criterion(ny, next_state_predict)
            self.model_nn_optim.zero_grad()
            model_loss.backward()
            self.model_nn_optim.step()
            # The loop will be teiminated while the average loss < limit
            if model_loss.data / self.batch_size < self.model_nn_error_limit:
                break
        # endregion


        # 循环更新actor网络和critic网路
        while True:

            # region update critic nn
            q_value = self.critic_nn(torch.cat((y, y_star), dim=1))


            next_q_value = self.critic_nn(torch.cat((ny, y_star), dim=1))
            target_q = penalty + self.gamma * next_q_value

            # 定义TD loss
            critic_loss = self.critic_criterion(q_value, target_q)
            self.critic_nn_optim.zero_grad()

            critic_loss.backward()
            self.critic_nn_optim.step()

            # endregion

            # region update actor nn

            # find u*
            best_u = self.find_best_u(u0=action, y=y, y_star=y_star)
            best_u = torch.FloatTensor(best_u)

            x = torch.cat((y, y_star), dim=1)

            # calculate current u
            cur_u = self.actor_nn(x)
            act_loss = self.actor_criterion(cur_u, best_u)
            self.actor_nn_optim.zero_grad()

            # optimize actor network
            act_loss.backward()
            self.actor_nn_optim.step()

            # act_loss足够小时才结束
            if act_loss / self.batch_size > self.actor_nn_error_limit:
                continue
            # 计算critic网络更新后的预测的V(k-1)值
            new_q_value = self.critic_nn(torch.cat((y, y_star), dim=1))
            diff_V = self.critic_criterion(q_value, new_q_value)

            # 两次V的输出足够小时才跳出
            if diff_V.data / self.batch_size < self.critic_nn_error_limit:
                break
            # endregion

    # 自己看论文公式18
    def find_best_u(self, u0,y,y_star):
        if self.u_begin is not None:
            u0 = u0.zero_() + torch.FloatTensor(self.u_begin)
        U = np.diag(self.u_bounds[:,1] - self.u_bounds[:,0])
        U = torch.FloatTensor(U)
        S = self.env.penalty_calculator.S
        S = torch.FloatTensor(S)
        u_mid = torch.FloatTensor(
            np.mean(self.u_bounds,axis=1)
        )

        # 我对论文的理解是用迭代的方法求u*
        for _ in itertools.count():

            tmp_u0 = u0.clone()
            # region 方向传播计算V对u的梯度

            u0.requires_grad = True
            self.critic_nn_optim.zero_grad()
            self.model_nn_optim.zero_grad()
            x_pred = self.model_nn(torch.cat((y, u0), dim=1))
            v_pred = self.critic_nn(torch.cat((x_pred, y_star), dim=1))
            v_pred.backward()
            # endregion
            u0_grad = u0.grad

            tmp = F.linear(u0_grad, -0.5*self.gamma*(U.mul(S).inverse().t()))
            tmp = torch.tanh(tmp)
            tmp = F.linear(tmp, U, bias=u_mid)
            u0 = tmp
            if (tmp_u0 - u0).norm()<0.1:
                break
            if _ > 50:
                print("Too much times for find u*")
                break


        return u0


    def predict(self, state, act):
        cur_y=state[2:4]
        x = torch.FloatTensor(np.hstack([cur_y, act]))
        return self.model_nn.forward(x)



    """
    """

    def cal_training_data(self):
        """

        :return:
        """
        self.env.reset()

        # 写在json里暂存，防止每次都靠仿真模型太慢
        json_path = "training_data_" + str(self.predict_training_rounds) + '.json'
        if os.path.exists(json_path):
            with open(json_path, 'r',) as fp:
                train_x, train_y = json.load(fp)
                train_x = np.array(train_x)
                train_y = np.array(train_y)
                return train_x, train_y

        train_x = []
        train_y = []
        # 生成训练数据
        print("模拟生成")
        for _ in range(self.predict_training_rounds):
            print(_)
            state = self.env.observation()[2:4]
            act = np.random.uniform(self.u_bounds[:,0], self.u_bounds[:,1])

            train_x.append(np.hstack([state,act])[np.newaxis,:])
            self.env.step(act)
            new_state = self.env.observation()[2:4]
            train_y.append(new_state[np.newaxis, :])
            if random.random() < 0.001:
                self.env.reset()
        # 写json暂存
        with open(json_path, 'w',) as fp:
            tmp_x = np.copy(train_x).tolist()
            tmp_y = np.copy(train_y).tolist()
            json.dump((tmp_x, tmp_y), fp)

        return train_x, train_y

    def test_predict_model(self, test_rounds=1000):
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
            pred_y_list.append(pred_y.detach().numpy() - old_y)
            self.env.step(act)
            real_y_list.append(self.env.observation()[2:4] - old_y)
        real_y_array = np.array(real_y_list)
        pred_y_array = np.array(pred_y_list)
        for i in range(self.env.size_yudc[0]):
            plt.plot(real_y_array[:,i])
            plt.plot(pred_y_array[:,i])
            plt.legend(['real','predict'])
            plt.show()

    def cal_predict_mse(self, test_rounds=1000, diff=False):

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
                old_y = old_y * 0
            pred_y_list.append(pred_y.detach().numpy() - old_y)
            self.env.step(act)
            real_y_list.append(self.env.observation()[2:4] - old_y)
        real_y_array = np.array(real_y_list)
        pred_y_array = np.array(pred_y_list)

        mse_li = []
        for i in range(self.env.size_yudc[0]):
            mse = mean_squared_error(real_y_array[:,i], pred_y_array[:,i])
            mse_li.append(mse)

        return mse_li




    def define_identification_nn(self, Vm, Lm):
        self.model_nn = Model(dim_in=self.env.size_yudc[0]+self.env.size_yudc[1],
                              dim_out=self.env.size_yudc[0],dim_hidden=4,device=self.device,Vm=Vm,Lm=Lm)


    def train_identification_model(self, Vm, Lm):
        """

        代码参考:
        https://www.pytorchtutorial.com/3-6-optimizer/#i
        :param Vm:
        :param Lm:
        :return:
        """
        # 定义预测模型
        self.define_identification_nn(Vm,Lm)
        self.model_nn_optim = torch.optim.Adam(self.model_nn.parameters(), lr=0.01,betas=(0.9,0.99))
        self.model_criterion = torch.nn.MSELoss()

        train_x, train_y = self.cal_training_data()
        torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))

        # 建立数据集
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.predict_batch_size,shuffle=True)

        mse_list = []
        # 训练预测模型
        for epoch in itertools.count():

            print("Epoch:{}".format(epoch+1), "")
            sum_loss = 0
            for step,(batch_x, batch_y) in enumerate(loader):
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                output = self.model_nn(b_x)
                loss = self.model_criterion(output,b_y)
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
            if sum_loss < self.model_nn_error_limit or epoch >= 50:
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


    def actor_nn_init(self,Va,La):
        """
        定义动作网络相关
        :param Va:
        :param La:
        :return:
        """
        self.actor_nn = Actor(dim_in=self.env.size_yudc[0]*2,dim_out=self.env.size_yudc[1],
                              device=self.device, dim_hidden=4,Va=Va,La=La)
        self.actor_nn_optim = torch.optim.Adam(self.actor_nn.parameters(), lr=0.3,betas=(0.9,0.99))
        self.actor_criterion = torch.nn.MSELoss()

    def critic_nn_init(self,Vc,Lc):
        """
        定义值函数评价网络相关
        :param Vc:
        :param Lc:
        :return:
        """
        self.critic_nn = Critic(dim_in=self.env.size_yudc[0]*2,
                                device=self.device, dim_out=1, dim_hidden=4,Vc=Vc,Lc=Lc)
        self.critic_nn_optim = torch.optim.Adam(self.critic_nn.parameters(), lr=0.05,betas=(0.9,0.99))
        self.critic_criterion = torch.nn.MSELoss()


    def plt_list(self):
        """
        训练结束后返回控制效果和网络收敛效果
        :return:
        """

        return [
            self.u0_plt,
            self.u1_plt,
            self.y0_plt,
            self.y1_plt,
            self.wa_plt,
            self.wm_plt,
            self.wc_plt
        ]


if __name__ == '__main__':

    env = Flotation(normalize=False)
    controller = ILPL(env=env,
                      u_bounds=env.u_bounds,
                      Vm=np.diag([0.1,1,1,0.1]),
                      Lm=np.diag([0.1,1])
                      )
