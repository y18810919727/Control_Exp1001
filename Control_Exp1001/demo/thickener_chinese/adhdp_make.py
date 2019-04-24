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
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
def new_adhdp(capacity=2, off_policy=False):
    replay_hdp = ReplayBuffer(capacity=capacity)
    env_ADHDP = Thickener()
    #exploration = No_Exploration()
    exploration = EGreedy(env_ADHDP.external_u_bounds, epsilon_start=0.5,epsilon_final=0,epsilon_decay=1000)
    adhdp = ADHDP(
        replay_buffer = replay_hdp,
        u_bounds = env_ADHDP.u_bounds,
        #exploration = None,
        exploration = exploration,
        env=env_ADHDP,
        gamma=0.7,

        batch_size = capacity,
        predict_batch_size=32,

        critic_nn_error_limit = 0.02,
        actor_nn_error_limit = 0.001,

        actor_nn_lr = 0.01,
        critic_nn_lr = 0.01,

        indice_y = None,
        indice_y_star = None,
        indice_c=None,
        hidden_critic = 6,
        hidden_actor = 6,
        max_iter_c = 50,
        off_policy=off_policy,
    )
    return adhdp
