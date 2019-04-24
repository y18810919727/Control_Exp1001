#!/usr/bin/python
import os
import json
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.optim as optim
from Control_Exp1001.control.base_control import ControlBase
from Control_Exp1001.simulation.flotation import Flotation
import itertools

from sklearn.metrics import mean_squared_error


class ILPL(ControlBase):
    def __init__(self, u_bounds=None,
                 env=None,
                 predict_training_rounds=10000,
                 Vm=None,
                 Lm=None,
                 gpu_id=1,
                 batch_size = 32,
                 model_nn_error_limit = 1.0

                 ):
        super(ILPL, self).__init__(u_bounds)
        if env is None:
            env = Flotation()
        self.env=env
        self.predict_training_rounds = predict_training_rounds

        self.device = None
        self.cude_device(gpu_id)
        self.batch_size = batch_size

        self.predict_training_losses = []
        self.model_nn = None
        self.model_nn_error_limit = model_nn_error_limit

        # Train model neural network
        self.train_identification_model(Vm=Vm,Lm=Lm)
        self.test_predict_model(test_rounds=400)




    def cude_device(self, cuda_id):
        use_cuda = torch.cuda.is_available()
        cuda = 'cuda:'+str(cuda_id)
        self.device = torch.device(cuda if use_cuda else "cpu")

    def _act(self, state):
        raise NotImplementedError


    def _train(self, s, u, ns, r, done):
        raise NotImplementedError


    def predict(self, state, act):
        cur_y=state[2:4]
        x = torch.FloatTensor(np.hstack([cur_y, act]))
        return self.model_nn.forward(x)




    """
    代码参考:
         https://www.pytorchtutorial.com/3-6-optimizer/#i
    """

    def cal_training_data(self):

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
        with open(json_path, 'w',) as fp:
            tmp_x = np.copy(train_x).tolist()
            tmp_y = np.copy(train_y).tolist()
            json.dump((tmp_x, tmp_y), fp)

        return train_x, train_y



    def test_predict_model(self, test_rounds=1000):
        self.env.reset()
        pred_y_list = []
        real_y_list = []
        #pred_y_list.append(self.env.observation()[2:4][np.newaxis,:])
        for _ in range(test_rounds):
            act = np.random.uniform(self.u_bounds[:,0],
                                    self.u_bounds[:,1])
            pred_y = self.predict(self.env.observation(), act)

            pred_y_list.append(pred_y.detach().numpy())
            self.env.step(act)
            real_y_list.append(self.env.observation()[2:4])
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




    def train_identification_model(self, Vm, Lm):
        self.model_nn = Model(dim_in=4,dim_out=2,dim_hidden=4,device=self.device,Vm=Vm,Lm=Lm)
        self.model_nn_optim = torch.optim.Adam(self.model_nn.parameters(), lr=0.01,betas=(0.9,0.99))
        self.critic = torch.nn.MSELoss()

        train_x, train_y = self.cal_training_data()
        torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))

        loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.batch_size,shuffle=True)
        mse_list = []

        for epoch in itertools.count():

            print("Epoch:{}".format(epoch+1), "")
            sum_loss = 0
            for step,(batch_x, batch_y) in enumerate(loader):
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                output = self.model_nn(b_x)
                loss = self.critic(output,b_y)
                self.model_nn_optim.zero_grad()
                loss.backward()

                self.model_nn_optim.step()
                self.predict_training_losses.append(loss.item())
                sum_loss += loss.item()


            print("Loss:{}".format(sum_loss))
            # 每过20轮评估一次mse
            if epoch % 20 == 0:
                mse_list.append(self.cal_predict_mse())
            if sum_loss < self.model_nn_error_limit or epoch > 50:
                break



        # 最后再评估一次
        mse_list.append(self.cal_predict_mse())
        # 绘制损失变化
        plt.figure()

        plt.title("Loss in various epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(self.predict_training_losses)
        plt.show()


        # 绘制预测mse变化
        plt.figure()
        mse_array = np.array(mse_list)
        for i in range(mse_array.shape[1]):
            plt.plot(mse_array[:,i])
            plt.plot(mse_array[:,i])
        plt.legend(['y1','y2'])
        plt.show()
        # 打印mse
        print(mse_array)

















class Model(torch.nn.Module):
    def __init__(self,dim_in,dim_out,device,Vm=None,Lm=None,dim_hidden=4,):
        super(Model,self).__init__()

        self.Vm=torch.nn.Linear(dim_in,dim_hidden,bias=False)
        if Vm is not None:
            self.Vm.weight.data=torch.FloatTensor(Vm)
            self.Vm.weight.requires_grad = False

        self.Wm = torch.nn.Linear(dim_hidden,dim_out)

        self.Lm = torch.nn.Linear(dim_out,dim_out,bias=False)
        self.Lm.weight.requires_grad = False

        if Lm is None:
            self.Lm.weight.data=torch.FloatTensor(np.diag(np.ones()))
        else:
            Lm = torch.FloatTensor(Lm)
            self.Lm.weight.data=torch.FloatTensor(Lm.inverse())

        self.to(device)

    def forward(self, x):
        y = self.Vm(x)
        y = torch.tanh(y)
        y = self.Wm(y)
        y = self.Lm(y)
        #z = torch.index_select(x,-1,torch.LongTensor([0,1]))
        return y


