#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json


from Control_Exp1001.demo.thickener.adhdp import ADHDP
from Control_Exp1001.simulation.thickener import Thickener
from Control_Exp1001.common.penaltys.demo_penalty import DemoPenalty
import matplotlib.pyplot as plt
from Control_Exp1001.demo.thickener.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.one_round_evaluation import OneRoundEvaluation
from Control_Exp1001.common.action_noise.e_greedy import EGreedy
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
replay_hdp = ReplayBuffer(capacity=20)
env_ADHDP = Thickener()
exploration = EGreedy(epsilon_start=1, epsilon_final=0.0001, epsilon_decay=300,action_bounds=env_ADHDP.u_bounds)

adhdp = ADHDP(
    replay_buffer = replay_hdp,
    u_bounds = env_ADHDP.u_bounds,
    #exploration = None,
    exploration = exploration,
    env=env_ADHDP,
    gamma=0.1,

    batch_size = 10,
    predict_batch_size=32,

    critic_nn_error_limit = 0.02,
    actor_nn_error_limit = 0.001,

    actor_nn_lr = 0.05,
    critic_nn_lr = 0.1,

    indice_y = None,
    indice_y_star = None,
    indice_c=None,
    hidden_critic = 10,
    hidden_actor = 10,
)
