#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json


import torch
from Control_Exp1001.control.base_ac import ACBase


class HDP(ACBase):
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

                 indice_y = None,
                 indice_y_star = None,
                 u_first = None
                 ):

        super(HDP, self).__init__(gpu_id=gpu_id, replay_buffer=replay_buffer,
                                  u_bounds=u_bounds, exploration=exploration)

        self.env=env

        self.device = None
        self.cuda_device(gpu_id)


        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size


        self.indice_c = [6, 7]

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

        self.y0_plt.push("ht*", state[0])
        self.y1_plt.push("Cu*", state[1])
        self.y0_plt.push("ht", state[2])
        self.y1_plt.push("Cu", state[3])
        # 第一次控制不用actor模型输出，否则会很离谱
        if self.first_act:

            self.u0_plt.push("fu", self.u_first[0])
            self.u1_plt.push("ff", self.u_first[1])
            self.first_act = False
            act = self.u_first

        # 用actor网络计算输出
        else:
            y = state[self.indice_y]
            y_star = state[self.indice_y_star]
            c = state[self.indice_c]

            x = torch.FloatTensor(np.hstack((y, y_star,c))).unsqueeze(0)
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
