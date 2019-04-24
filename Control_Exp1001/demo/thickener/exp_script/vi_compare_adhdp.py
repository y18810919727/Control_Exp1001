#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json

from Control_Exp1001.simulation.thickener import Thickener
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
from Control_Exp1001.common.action_noise.e_greedy import EGreedy
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
from Control_Exp1001.common.action_noise.gaussian_noise import GaussianExploration
from Control_Exp1001.demo.thickener.controllers.value_iterate import VI
from Control_Exp1001.demo.thickener.controllers.adhdp import ADHDP
import torch
import random
from Control_Exp1001.common.penaltys.quadratic import Quadratic
import matplotlib.pyplot as plt

from Control_Exp1001.demo.thickener.common.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.common.one_round_evaluation import OneRoundEvaluation


penalty_para = {
    #"weight_matrix": [0, 0.004],
    "weight_matrix": [0, 0.004],
    "S": [0.0001, 0.0008],
    #"S": [0.0003, 0.0024],
    #"S": [1.3, 2.4],
    #"S": [0.0000, 0.000],
}
thickner_para = {
    "dt":1,
    "noise_in": False,
    "noise_p": 0.002,
    #"noise_type": "const",
    "noise_type": 3,
    'time_length': 20,# 浓密机每次仿真20秒
}
from Control_Exp1001.demo.thickener.common import exp_name
exp_name.set_exp_name('VIandADHDP')
EXP_NAME = exp_name.get_exp_name()
img_path = os.path.join('../images',EXP_NAME)
if not os.path.exists(img_path):
    os.mkdir(img_path)
def new_vi():
    capacity=2
    predict_round=3000
    u_optim='sgd'
    gamma=0.6
    replay_vi = ReplayBuffer(capacity=capacity)
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

        actor_nn_lr = 0.5,
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
        img_path=EXP_NAME,
        test_period=1000,

    )
    env_VI.reset()
    vi.train_identification_model()
    #vi.test_predict_model(test_rounds=100)
    return vi

def new_adhdp(random_act=False):
    period = 20
    capacity = period
    train_period= period
    batch_size= period
    off_policy = False
    replay_hdp = ReplayBuffer(capacity=capacity)
    env_ADHDP = Thickener()
    #exploration = No_Exploration()
    #exploration = EGreedy(env_ADHDP.external_u_bounds, epsilon_start=0.6,epsilon_final=0,epsilon_decay=10)
    exploration = GaussianExploration(action_bounds=env_ADHDP.external_u_bounds,
                                       min_sigma=0.00, max_sigma=0.01, decay_period=600)
    if random_act:
        exploration = EGreedy(action_bounds=env_ADHDP.external_u_bounds,
                              epsilon_start=1,epsilon_final=1,epsilon_decay=100)
        train_period = 20
    adhdp = ADHDP(
        replay_buffer = replay_hdp,
        u_bounds = env_ADHDP.u_bounds,
        #exploration = None,
        exploration = exploration,
        env=env_ADHDP,
        gamma=0.8,

        batch_size = batch_size,
        predict_batch_size=32,

        critic_nn_error_limit = 0.05,
        actor_nn_error_limit = 0.001,

        actor_nn_lr = 0.003,
        critic_nn_lr = 0.05,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_critic = 16,
        hidden_actor = 20,
        off_policy=off_policy,
        Nc=1000,
        Na=50,
        train_period=train_period,
        test_period=1
    )
    return adhdp

def run_vi(rounds=1000,seed=random.randint(0,1000000),name='VI',capacity=2,
           predict_round=3000,u_optim='adam',):

    print('seed :',seed)
    torch.manual_seed(seed)
    vi_para = {
        'gamma': 0.2
    }
    vi = new_vi()
    penalty = Quadratic(**penalty_para)
    env_vi = Thickener(
        penalty_calculator=penalty,
        **thickner_para,
    )

    res1 = OneRoundExp(controller=vi, env=env_vi,max_step=rounds, exp_name=name).run()
    print(name,':',vi.u_iter_times*1.0/rounds)

    return res1


def run_adhdp(rounds=1000,seed=random.randint(0,1000000),name='ADHDP', predict_round=800, random_act=False):
    print('seed :',seed)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    adhdp = new_adhdp(random_act=random_act)
    penalty = Quadratic(**penalty_para)
    env_hdp = Thickener(
        penalty_calculator=penalty,
        random_seed=seed,
        **thickner_para,
    )

    res1 = OneRoundExp(controller=adhdp, env=env_hdp,max_step=rounds, exp_name=name).run()
    return res1


if __name__ == '__main__':

    round = 1600
    predict_round=800
    res_list = []
    rand_seed = np.random.randint(0,10000000)
    # rand_seed = 320743
    #rand_seed = 6251962
    #rand_seed = 9992362
    #rand_seed = 2983269
    #res_list.append(run_vi(rounds=round,seed=rand_seed, name='HCNVI', predict_round=predict_round))
    res_list.append(run_adhdp(rounds=round,seed=rand_seed, name='ADHDP', predict_round=predict_round))
    #res_list.append(run_adhdp(rounds=round,seed=rand_seed, name='rand policy', predict_round=predict_round,random_act=True))
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()

