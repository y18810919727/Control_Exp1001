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
from Control_Exp1001.demo.thickener.controllers.value_iterate_ub import VIub
from Control_Exp1001.demo.thickener.controllers.hdp import HDP
import torch
import random
from Control_Exp1001.common.penaltys.quadratic import Quadratic
import matplotlib.pyplot as plt

from Control_Exp1001.demo.thickener.common.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.common.one_round_evaluation import OneRoundEvaluation

penalty_para = {
    #"weight_matrix": [0, 0.002],
    "weight_matrix": [0, 0.004],
    "S": [0.0001, 0.0008],
    #"S": [0.0003, 0.0024],
    #"S": [0.0000, 0.000],
}
thickner_para = {
    "dt":1,
    "noise_in": False,
    "noise_p": 0.002,
    "noise_type": 1,
    'time_length': 20,# 浓密机每次仿真20秒
}
from Control_Exp1001.demo.thickener.common import exp_name
exp_name.set_exp_name('VIubtimes')
EXP_NAME = exp_name.get_exp_name()
img_path = os.path.join('../images',EXP_NAME)
if not os.path.exists(img_path):
    os.mkdir(img_path)
def new_vi_ub(find_times=20, find_lr=0.05):
    capacity=2
    predict_round=3000
    u_optim='SGD'
    replay_vi = ReplayBuffer(capacity=capacity)
    env_VI = Thickener(
        noise_p=0.03,
        noise_in=True,
    )
    exploration = No_Exploration()

    print('make new viuk controller')
    vi = VIub(
        replay_buffer = replay_vi,
        u_bounds = env_VI.u_bounds,
        #exploration = None,
        exploration = exploration,
        env=env_VI,
        predict_training_rounds=predict_round,
        gamma=0.6,

        batch_size = capacity,
        predict_batch_size=32,

        model_nn_error_limit = 0.0008,
        critic_nn_error_limit = 0.001,
        actor_nn_error_limit = 0.001,

        actor_nn_lr = 0.005,
        critic_nn_lr = 0.01,
        model_nn_lr = 0.01,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_model = 10,
        hidden_critic = 14,
        hidden_actor = 14,
        predict_epoch= 30,
        u_optim=u_optim,
        find_lr= find_lr,
        find_time_max=find_times

    )
    env_VI.reset()
    vi.train_identification_model()
    return vi

def run_vi_ub(rounds=1000,seed=random.randint(0,1000000),name='VI_uk',capacity=2,
              predict_round=3000,u_optim='sgd',find_times=20):

    print('seed :',seed)
    torch.manual_seed(seed)
    vi = new_vi_ub(find_times=find_times)
    penalty = Quadratic(**penalty_para)
    env_vi = Thickener(
        penalty_calculator=penalty,
        **thickner_para,
    )

    res1 = OneRoundExp(controller=vi, env=env_vi,max_step=rounds, exp_name=name).run()
    print(name,':',vi.u_iter_times*1.0/rounds)

    return res1

if __name__ == '__main__':

    round = 800
    predict_round=800
    res_list = []
    rand_seed = np.random.randint(0,10000000)
    # rand_seed = 320743
    for u_times in range(5,20,10):
        res_list.append(run_vi_ub(rounds=round,seed=rand_seed, name='VIUB-find='+str(u_times), predict_round=predict_round,find_times=u_times))
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()

