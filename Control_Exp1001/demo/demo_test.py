#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
from Control_Exp1001.common.action_noise.gaussian_noise import GaussianExploration
from matplotlib import pyplot  as plt
from Control_Exp1001.exp.base_exp import BaseExp
from Control_Exp1001.exp.exp_perform import exp_multi_thread_run
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
from Control_Exp1001.common.penaltys.demo_penalty import DemoReward
from Control_Exp1001.simulation.demo_simmulation import DemoSimulation as Env
from Control_Exp1001.control.td3 import Td3
from Control_Exp1001.control.demo_control import DemoControl
from Control_Exp1001.common.evaluation.base_evaluation import EvaluationBase
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
if __name__ == '__main__':
    if __name__ == '__main__':
        reward = DemoReward(weight_matrix=[1, 10])
        env = Env(
            dt=1,
            size_yudc=[2, 2, 0, 2],
            y_low=[-15, -15],
            y_high=[15, 15],
            u_high=[2, 2],
            u_low=[-2, -2],
            reward_calculator=reward,
            normalize=False
        )
        controller = DemoControl(u_bounds=env.external_u_bounds)


        exp = BaseExp(
            env=env,
            controller=controller,
            max_frame=100000,
            rounds=10,
            max_step=10,
            eval_rounds=5,
            eval_length=None,
            exp_name="exploration_noise=1.1"
        )

        # exp.render_mode = True
        # controller2.render_mode = True
        # env2.render_mode = True
        # exp_res1 = exp1.run()
        # exp_res2 = exp2.run()
        # res_list = exp_multi_thread_run([exp1, exp2, exp3])
        res_list = exp_multi_thread_run([exp])
        evaluation = EvaluationBase(res_list=res_list,
                                    training_rounds=exp.rounds,
                                    exp_name=[ "best control"],
                                    y_name=["height", "concentration"],
                                    y_num=2,
                                    reward_plt_param={"figsize": (10, 5)},
                                    eval_plt_param={"figsize": (10, 8)}
                                    )
        evaluation.draw_rewards()
        evaluation.draw_eval()
        evaluation.error_eval()