#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
from multiprocessing import Process
import json
import torch
from Control_Exp1001.demo.thickener.hdp import HDP
from Control_Exp1001.simulation.thickener import Thickener
from Control_Exp1001.common.penaltys.quadratic import Quadratic
import matplotlib.pyplot as plt
from Control_Exp1001.demo.thickener.one_round_exp import OneRoundExp
from Control_Exp1001.demo.thickener.one_round_evaluation import OneRoundEvaluation
from Control_Exp1001.demo.thickener.adhdp_make import adhdp
import random
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
# 效用函数参数
penalty_para = {
    #"weight_matrix": [0, 0.002],
    "weight_matrix": [0, 0.004],
    #"S": [0.00001, 0.00008],
    "S": [0.0001, 0.0008]
}
# 仿真过程浓密机参数
thickner_para = {
    "dt":1,
    "noise_in": False,
    "noise_p": 0.002,
    "noise_type": 1,
}


def run_hdp(rounds=1000,seed=random.randint(0,1000000),name='HDP', predict_round=800):

    print('seed :',seed)
    from Control_Exp1001.demo.thickener.hdp_maker import new_hdp
    hdp = new_hdp(predict_round=predict_round)
    penalty = Quadratic(**penalty_para)
    env_hdp = Thickener(
                    penalty_calculator=penalty,
                    **thickner_para,
                    )


    res1 = OneRoundExp(controller=hdp, env=env_hdp,max_step=rounds, exp_name=name).run()


    return res1



    #controller.test_predict_model(test_rounds=100)


def run_hdp_sample(rounds=1000,seed=random.randint(0,1000000)):


    print('seed :',seed)
    print('hdp_sample')
    from Control_Exp1001.demo.thickener.hdp_sample_maker import hdp_sample
    penalty = Quadratic(**penalty_para)
    env_hdp = Thickener(
            penalty_calculator=penalty,
            **thickner_para,
    )

    res1 = OneRoundExp(controller=hdp_sample, env=env_hdp,max_step=rounds, exp_name='HDP_sample').run()

    return res1

    #controller.test_predict_model(test_rounds=100)


def run_vi(rounds=1000,seed=random.randint(0,1000000),name='VI',capacity=2,
           predict_round=3000):

    print('seed :',seed)
    torch.manual_seed(seed)
    from Control_Exp1001.demo.thickener.vi_maker import new_vi
    vi = new_vi(capacity=capacity,predict_round=predict_round)
    penalty = Quadratic(**penalty_para)
    env_vi = Thickener(
                    penalty_calculator=penalty,
                    **thickner_para,
                    )


    res1 = OneRoundExp(controller=vi, env=env_vi,max_step=rounds, exp_name=name).run()


    return res1



def vi_compare_hdp():
    round = 400
    predict_round=800
    res_list = []
    rand_seed = np.random.randint(0,10000000)
    res_list.append(run_vi(rounds=round,seed=rand_seed, name='VI', predict_round=predict_round))
    res_list.append(run_hdp(rounds=round,seed=rand_seed, name='HDP', predict_round=predict_round))
    eval_res = OneRoundEvaluation(res_list=res_list)
    eval_res.plot_all()


if __name__ == '__main__':

    # 比较纯vi和hdp区别
    vi_compare_hdp()



