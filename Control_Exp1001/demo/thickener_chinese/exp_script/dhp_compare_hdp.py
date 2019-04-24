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
from Control_Exp1001.demo.thickener.controllers.hdp import HDP
from Control_Exp1001.demo.thickener.controllers.dhp import DHP
import torch
import random
from Control_Exp1001.common.penaltys.quadratic import Quadratic
import matplotlib.pyplot as plt

from Control_Exp1001.demo.thickener.common.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.common.one_round_evaluation import OneRoundEvaluation

penalty_para = {
    #"weight_matrix": [0, 0.002],
    "weight_matrix": [0, 0.006],
    "S": [0.0001, 0.0008],
    #"S": [0.0003, 0.0024],
    #"S": [0.0000, 0.000],
}
thickner_para = {
    "dt":1,
    "noise_in": False,
    "noise_p": 0.002,
    #"noise_type": 1,
    "noise_type": "None",
    'time_length': 20,# 浓密机每次仿真20秒
}
from Control_Exp1001.demo.thickener.common import exp_name

exp_name.set_exp_name('HDPandDHP')
EXP_NAME = exp_name.get_exp_name()
img_path = os.path.join('../images',EXP_NAME)
if not os.path.exists(img_path):
    os.mkdir(img_path)
def new_dhp():
    capacity= 20
    predict_round=6000
    gamma=0.0
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

        batch_size = 20,
        predict_batch_size=32,

        model_nn_error_limit = 0.0008,
        critic_nn_error_limit = 0.01,
        actor_nn_error_limit = 0.001,

        # 0.005
        actor_nn_lr = 0.005,
        critic_nn_lr = 0.001,
        model_nn_lr = 0.01,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_model = 10,
        hidden_critic = 12,
        hidden_actor = 14,
        predict_epoch= 30,
        Na=220,
        Nc=100,
        test_period=1,
        img_path=EXP_NAME,
    )
    env_DHP.reset()
    dhp.train_identification_model()
    return dhp

def new_hdp():
    predict_round=3000
    gamma=0.6
    replay_hdp = ReplayBuffer(capacity=2)
    env_HDP = Thickener(
        noise_p=0.01,
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
        actor_nn_lr = 0.002,
        critic_nn_lr = 0.01,
        model_nn_lr = 0.01,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_model = 10,
        hidden_critic = 14,
        hidden_actor = 14,
        predict_epoch= 30,
        Na=220,
        img_path=EXP_NAME
    )
    env_HDP.reset()
    hdp.train_identification_model()
    return hdp

def run_dhp(rounds=1000,seed=random.randint(0,1000000),name='DHP',capacity=2,
           predict_round=3000,u_optim='adam',):

    print('seed :',seed)
    torch.manual_seed(seed)
    dhp_para = {
        #'gamma': 0.2
    }
    dhp = new_dhp()
    penalty = Quadratic(**penalty_para)
    env_dhp = Thickener(
        penalty_calculator=penalty,
        **thickner_para,
    )

    res1 = OneRoundExp(controller=dhp, env=env_dhp,max_step=rounds, exp_name=name).run()
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


if __name__ == '__main__':

    round = 1600
    predict_round=3000
    res_list = []
    rand_seed = np.random.randint(0,10000000)
    # rand_seed = 320743
    res_list.append(run_dhp(rounds=round,seed=rand_seed, name='DHP', predict_round=predict_round))
    res_list.append(run_hdp(rounds=round,seed=rand_seed, name='HDP', predict_round=predict_round))
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()

