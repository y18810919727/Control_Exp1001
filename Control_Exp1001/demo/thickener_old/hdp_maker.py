#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json


from Control_Exp1001.demo.thickener.hdp import HDP
from Control_Exp1001.simulation.thickener import Thickener
from Control_Exp1001.common.penaltys.demo_penalty import DemoPenalty
import matplotlib.pyplot as plt
from Control_Exp1001.demo.thickener.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.one_round_evaluation import OneRoundEvaluation
from Control_Exp1001.common.action_noise.gaussian_noise import GaussianExploration
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer

def new_hdp(predict_round=800, gamma=0.4):
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
    )
    env_HDP.reset()
    hdp.train_identification_model()
    return hdp
