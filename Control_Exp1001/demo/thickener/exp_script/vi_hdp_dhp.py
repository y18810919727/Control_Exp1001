#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json

from Control_Exp1001.simulation.thickener import Thickener
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
from Control_Exp1001.demo.thickener.controllers.value_iterate import VI
from Control_Exp1001.demo.thickener.controllers.hdp import HDP

from Control_Exp1001.demo.thickener.controllers.dhp_lijia import DHP
import torch
import random
from Control_Exp1001.common.penaltys.quadratic import Quadratic
import matplotlib.pyplot as plt

from Control_Exp1001.demo.thickener.common.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.common.one_round_evaluation import OneRoundEvaluation

penalty_para = {
    #"weight_matrix": [0, 0.002],
    #"weight_matrix": [0, 0.010],
    "weight_matrix": [0, 0.0040],
    "S": [0.0001, 0.0008],
    #"S": [0.0003, 0.0024],
    #"S": [0.0000, 0.000],
}
thickner_para = {
    "dt":1,
    "noise_in": False,
    "noise_p": 0.002,
    "noise_type": 2,
    'time_length': 20,# 浓密机每次仿真20秒
}
from Control_Exp1001.demo.thickener.common import exp_name
exp_name.set_exp_name('VIandHDPandDHP')
EXP_NAME = exp_name.get_exp_name()
img_path = os.path.join('../images',EXP_NAME)
if not os.path.exists(img_path):
    os.mkdir(img_path)
def new_vi():
    capacity=2 #经验池的大小，需要大于或等于batch_size
    predict_round=3000
    u_optim='sgd' # 寻找u*使用的梯度下降算法
    gamma=0.6
    replay_vi = ReplayBuffer(capacity=capacity)
    # 这个浓密机是用来生成数据训练预测模型用的
    env_VI = Thickener(
        noise_p=0.03,
        noise_in=True,
    )
    exploration = No_Exploration()

    print('make new vi controller')
    vi = VI(
        replay_buffer = replay_vi,
        u_bounds = env_VI.u_bounds,
        #exploration = None,
        exploration = exploration,
        env=env_VI,
        predict_training_rounds=predict_round,
        gamma=gamma,

        batch_size = capacity,
        predict_batch_size=32,

        model_nn_error_limit = 0.0008,
        critic_nn_error_limit = 0.001,
        actor_nn_error_limit = 0.001,

        actor_nn_lr = 0.005,
        critic_nn_lr = 0.02,
        model_nn_lr = 0.01,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_model = 10,
        hidden_critic = 14,
        hidden_actor = 14,
        predict_epoch= 30,
        Nc=500,
        u_optim=u_optim,
        img_path=EXP_NAME
    )
    env_VI.reset()
    vi.train_identification_model()
    #vi.test_predict_model(test_rounds=100)
    return vi

def new_hdp():
    predict_round=3000
    gamma=0.6
    replay_hdp = ReplayBuffer(capacity=2)
    env_HDP = Thickener(
        noise_p=0.03,
        noise_in=True,
    )
    exploration = No_Exploration()

    print('make new hdp controller')
    hdp = HDP(
        replay_buffer = replay_hdp,
        u_bounds = env_HDP.u_bounds,
        #exploration = None,
        exploration = exploration,
        env=env_HDP,
        predict_training_rounds=predict_round,
        gamma=gamma,

        batch_size = 2,
        predict_batch_size=32,

        model_nn_error_limit = 0.0008,
        critic_nn_error_limit = 0.001,
        actor_nn_error_limit = 0.001,

        # 0.005
        actor_nn_lr = 0.003,
        critic_nn_lr = 0.02,
        model_nn_lr = 0.01,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_model = 10,
        hidden_critic = 14,
        hidden_actor = 14,
        predict_epoch= 30,
        Na=220,
        Nc = 500,
        img_path=EXP_NAME
    )
    env_HDP.reset()
    hdp.train_identification_model()
    return hdp


def new_dhp():
    capacity=3
    predict_round=3000
    gamma=0.6
    replay_DHP = ReplayBuffer(capacity=capacity)
    env_DHP = Thickener(
        noise_p=0.03,
        noise_in=True,
    )
    exploration = No_Exploration()

    print('make new dhp controller')
    dhp = DHP(
        replay_buffer = replay_DHP,
        u_bounds = env_DHP.u_bounds,
        #exploration = None,
        exploration = exploration,
        env=env_DHP,
        predict_training_rounds=predict_round,
        gamma=gamma,

        batch_size = 2,
        predict_batch_size=32,

        model_nn_error_limit = 0.0008,
        critic_nn_error_limit = 0.001,
        actor_nn_error_limit = 0.0001,

        # 0.005
        actor_nn_lr = 0.008,
        critic_nn_lr = 0.01,
        model_nn_lr = 0.01,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_model = 10,
        hidden_critic = 12,
        hidden_actor = 14,
        predict_epoch= 30,
        Na=200,
        Nc=300,
        img_path=EXP_NAME
    )
    env_DHP.reset()
    dhp.train_identification_model()
    return dhp
def run_vi(rounds=1000,seed=random.randint(0,1000000),name='VI',capacity=2,
           predict_round=3000,u_optim='adam',):

    print('seed :',seed)
    torch.manual_seed(seed)
    vi_para = {
        'gamma': 0.2
    }
    vi = new_vi()
    penalty = Quadratic(**penalty_para)# 效用函数
    env_vi = Thickener(
        penalty_calculator=penalty,
        **thickner_para,
    )

    res1 = OneRoundExp(controller=vi, env=env_vi,max_step=rounds, exp_name=name).run()
    print(name,':',vi.u_iter_times*1.0/rounds)

    return res1

def run_hdp(rounds=1000,seed=random.randint(0,1000000),name='HDP', predict_round=800):
    print('seed :',seed)
    hdp_para = {
        'gamma':0.2
    }
    hdp = new_hdp()
    penalty = Quadratic(**penalty_para)
    env_hdp = Thickener(
        penalty_calculator=penalty,
        **thickner_para,
    )

    res1 = OneRoundExp(controller=hdp, env=env_hdp,max_step=rounds, exp_name=name).run()
    return res1

def run_dhp(rounds=800,seed=random.randint(0,1000000),name='DHP',capacity=2,
            predict_round=3000,u_optim='adam',):

    print('seed :',seed)
    torch.manual_seed(seed)
    dhp = new_dhp()
    penalty = Quadratic(**penalty_para)
    env_dhp = Thickener(
        penalty_calculator=penalty,
        **thickner_para,
    )

    res1 = OneRoundExp(controller=dhp, env=env_dhp,max_step=rounds, exp_name=name).run()
    return res1

if __name__ == '__main__':

    round = 1600
    predict_round=800
    res_list = []
    rand_seed = np.random.randint(0,10000000)
    rand_seed = 320743
    res_list.append(run_hdp(rounds=round,seed=rand_seed, name='HDP', predict_round=predict_round))
    res_list.append(run_dhp(rounds=round,seed=rand_seed, name='DHP', predict_round=predict_round))
    res_list.append(run_vi(rounds=round,seed=rand_seed, name='HCNVI', predict_round=predict_round))
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()
