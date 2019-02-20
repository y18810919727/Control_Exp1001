#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
from Control_Exp1001.simulation.flotation import Flotation
from Control_Exp1001.demo.flotation.controller import ILPL
from Control_Exp1001.demo.flotation.flotation_exp import FlotationExp
from Control_Exp1001.exp.exp_perform import exp_multi_thread_run
from Control_Exp1001.common.evaluation.base_evaluation import EvaluationBase
from Control_Exp1001.common.penaltys.integral_penalty import IntegralPenalty
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer

from Control_Exp1001.simulation.simulation_test import simulation_test





def run():

    # 定义积分惩罚项
    penalty = IntegralPenalty(weight_matrix=[1,1], S=[0.1,0.1])
    #penalty = IntegralPenalty(weight_matrix=[1,1], S=[0.00001,0.00001])
    # 定义初始化env对象的参数
    env_para = {
        "dt": 20,
        "normalize": False,
        "penalty_calculator": penalty,
        "one_step_length": 0.005,
        #"y_star": np.array([17.32, 0.84], dtype=float)
        "y_star": np.array([17.3, 0.8], dtype=float)
    }


    # 验证env_para情况下系统收敛状况
    # Flotation.flotation_test(init_para=env_para)
    # simulation_test(Flotation, init_para=env_para, mode="const", const_u=[[1.8, 19]])
    # return

    env = Flotation(
        **env_para
    )
    # 重新定义一下y_star
    env.y_star = np.array([17.32, 0.84])


    # 回放池大小为1，batch_size为1
    replaybuff = ReplayBuffer(capacity=1)
    # 参照论文给出的参数
    controller = ILPL(env=env,
                      u_bounds=env.u_bounds,
                      replay_buffer=replaybuff,
                      Vm=np.diag([0.1,1,1,0.1]),
                      Lm=np.diag([0.1,1]),
                      Va=np.diag([0.1, 1, 0.1, 1]),
                      La=np.diag([0.1,1]),
                      Vc=np.diag([0.1,1,0.1,1]),
                      Lc=np.diag([0.1]),
                      predict_training_rounds=5000,
                      gamma=0.6,
                      batch_size=1,
                      predict_batch_size = 32,
                      model_nn_error_limit=0.08,
                      critic_nn_error_limit=0.1,
                      actor_nn_loss= 0.6,
                      u_iter=30,
                      u_begin=[1.5,20],
                      indice_y=[2, 3],
                      indice_y_star=[0, 1],
                      )
    # 定义实验块
    exp = FlotationExp(
        env=env,
        controller=controller,
        max_step=200,
        exp_name="ILPL"
    )

    exp.run()
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
