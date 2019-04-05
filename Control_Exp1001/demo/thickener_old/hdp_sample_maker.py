#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json


from Control_Exp1001.demo.thickener.hdp_sample import HDP_sample
from Control_Exp1001.simulation.thickener import Thickener
from Control_Exp1001.common.penaltys.demo_penalty import DemoPenalty
import matplotlib.pyplot as plt
from Control_Exp1001.demo.thickener.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.one_round_evaluation import OneRoundEvaluation
from Control_Exp1001.common.action_noise.e_greedy import EGreedy
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
replay_hdp_sample = ReplayBuffer(capacity=30)
env_HDP_sample = Thickener(
    noise_p=0.01,
    noise_in=True
)
exploration = EGreedy(epsilon_start=0.0, epsilon_final=0.0000, epsilon_decay=100,action_bounds=env_HDP_sample.u_bounds)

hdp_sample = HDP_sample(
    replay_buffer = replay_hdp_sample,
    u_bounds = env_HDP_sample.u_bounds,
    #exploration = None,
    exploration = exploration,
    env=env_HDP_sample,
    predict_training_rounds=3000,
    gamma=0.1,

    batch_size = 10,
    predict_batch_size=32,

    model_nn_error_limit = 0.0008,
    critic_nn_error_limit = 0.02,
    actor_nn_error_limit = 0.5,

    actor_nn_lr = 0.05,
    critic_nn_lr = 0.1,
    model_nn_lr = 0.01,

    indice_y = None,
    indice_y_star = None,
    indice_c=None,
    hidden_model = 10,
    hidden_critic = 14,
    hidden_actor = 10,
    predict_epoch= 30,
)
env_HDP_sample.reset()
hdp_sample.train_identification_model()
