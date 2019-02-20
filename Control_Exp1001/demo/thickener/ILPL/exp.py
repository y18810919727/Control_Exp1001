#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
from Control_Exp1001.demo.thickener.ILPL.controller import ILPL
from Control_Exp1001.exp.one_round_exp import OneRoundExp
from Control_Exp1001.common.evaluation.one_round_evaluation import OneRoundEvaluation
from Control_Exp1001.exp.exp_perform import exp_multi_thread_run
from Control_Exp1001.common.evaluation.base_evaluation import EvaluationBase
from Control_Exp1001.common.penaltys.integral_penalty import IntegralPenalty
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer

from Control_Exp1001.simulation.simulation_test import simulation_test
from Control_Exp1001.simulation.thickener import Thickener




def run():

    # 定义积分惩罚项
    penalty = IntegralPenalty(weight_matrix=[200,0.02], S=[0.1,0.1])
    #penalty = IntegralPenalty(weight_matrix=[1,1], S=[0.00001,0.00001])
    # 定义初始化env对象的参数
    env_para = {
        "dt": 1,
        "normalize": False,
        "noise_in": False,
        "penalty_calculator": penalty,
        "y_star":[1.55, 650],
        "y_start":[1.4, 680]
        #"y_star": np.array([17.32, 0.84], dtype=float)
    }



    env = Thickener(
        **env_para
    )

    env.reset()
    # 回放池大小为1，batch_size为1
    replaybuff = ReplayBuffer(capacity=1)
    # 参照论文给出的参数
    controller = ILPL(env=env,
                      u_bounds=env.u_bounds,
                      replay_buffer=replaybuff,
                      Vm=np.diag([1, 0.01,0.1,0.1,0.1,0.1]),
                      Lm=np.diag([1, 0.01]),
                      Va=np.diag([1, 0.01, 1, 0.01, 0.1, 0.1]),
                      La=np.diag([1,1]),
                      Vc=np.diag([1, 0.01, 1, 0.01, 0.1, 0.1]),
                      Lc=np.diag([0.1]),
                      predict_training_rounds=5000,
                      gamma=0.6,
                      batch_size=1,
                      predict_batch_size = 32,
                      model_nn_error_limit=0.08,
                      critic_nn_error_limit=0.1,
                      actor_nn_loss= 0.6,
                      u_iter=30,
                      u_begin=[80, 38],
                      indice_y=[2, 3],
                      indice_y_star=[0, 1],
                      u_first=[80, 38]
                      )
    # 定义实验块
    exp = OneRoundExp(
        env=env,
        controller=controller,
        max_step=300,
        exp_name="ILPL"
    )

    res = exp.run()
    eval_res = OneRoundEvaluation(res_list=[res])
    eval_res.plot_all()
    # res_list = exp_multi_thread_run([exp])
    # evaluation = EvaluationBase(res_list=res_list,
    #                             training_rounds=exp.rounds,
    #                             exp_name=[exp.exp_name],
    #                             y_name=["Lcg", "Ltg"],
    #                             y_num=2,
    #                             penalty_plt_param={"figsize": (10, 5)},
    #                             eval_plt_param={"figsize": (10, 8)}
    #                             )
    # evaluation.draw_penaltys()
    # evaluation.draw_eval()
    # evaluation.error_eval()


if __name__ == '__main__':
    run()
