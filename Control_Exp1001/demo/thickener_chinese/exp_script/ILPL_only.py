#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json

from Control_Exp1001.demo.thickener_chinese.thickener_chinese import Thickener
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
from Control_Exp1001.demo.thickener_chinese.controllers.value_iterate import VI
from Control_Exp1001.demo.thickener_chinese.controllers.hdp import HDP

from Control_Exp1001.demo.thickener_chinese.controllers.dhp_lijia import DHP
from Control_Exp1001.demo.thickener_chinese.controllers.ILPL2 import ILPL
import torch
import random
from Control_Exp1001.common.penaltys.quadratic import Quadratic
import matplotlib.pyplot as plt

from Control_Exp1001.demo.thickener_chinese.common.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener_chinese.common.one_round_evaluation import OneRoundEvaluation

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
    "noise_type": 3,
    'time_length': 20,# 浓密机每次仿真20秒
}
from Control_Exp1001.demo.thickener_chinese.common import exp_name
exp_name.set_exp_name('ILPL')
EXP_NAME = exp_name.get_exp_name()
img_path = os.path.join('../images',EXP_NAME)
if not os.path.exists(img_path):
    os.mkdir(img_path)

def new_ILPL():
    predict_round=3000
    gamma=0.6
    replay_ILPL = ReplayBuffer(capacity=4)
    env_ILPL = Thickener(
        noise_p=0.03,
        noise_in=True,
    )
    exploration = No_Exploration()

    print('make new ilpl controller')
    ilpl = ILPL(
        replay_buffer = replay_ILPL,
        u_bounds = env_ILPL.u_bounds,
        #exploration = None,
        exploration = exploration,
        env=env_ILPL,
        predict_training_rounds=predict_round,
        gamma=gamma,

        batch_size = 2,
        predict_batch_size=32,

        model_nn_error_limit = 0.0008,
        critic_nn_error_limit = 0.1,
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
    env_ILPL.reset()
    ilpl.train_identification_model()
    return ilpl

def run_ILPL(rounds=1000,seed=random.randint(0,1000000),name='ILPL', predict_round=800):
    print('seed :',seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ilpl = new_ILPL()
    penalty = Quadratic(**penalty_para)
    env_ILPL = Thickener(
        penalty_calculator=penalty,
        **thickner_para,
    )

    res1 = OneRoundExp(controller=ilpl, env=env_ILPL,max_step=rounds, exp_name=name).run()
    return res1


if __name__ == '__main__':

    round = 1200
    predict_round=800
    res_list = []
    rand_seed = np.random.randint(0,10000000)
    #rand_seed = 212836
    #rand_seed = 320743
    #rand_seed = 1202966
    #rand_seed = 8254817 VI better DHP better HDP
    res_list.append(run_ILPL(rounds=round,seed=rand_seed, name='ILPL', predict_round=predict_round))
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()
