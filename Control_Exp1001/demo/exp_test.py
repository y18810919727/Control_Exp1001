
# -*- coding:utf8 -*-
from Control_Exp1001.common.action_noise.gaussian_noise import GaussianExploration
from Control_Exp1001.common.action_noise.no_exploration import No_Exploration
from Control_Exp1001.common.action_noise.e_greedy import EGreedy
from matplotlib import pyplot  as plt
from Control_Exp1001.exp.base_exp import BaseExp
from Control_Exp1001.exp.exp_perform import exp_multi_thread_run
from Control_Exp1001.common.replay.replay_buffer import ReplayBuffer
from Control_Exp1001.common.penaltys.demo_penalty import DemoPenalty
from Control_Exp1001.simulation.demo_simmulation import DemoSimulation as Env
from Control_Exp1001.control.td3 import Td3
from Control_Exp1001.control.demo_control import DemoControl
from Control_Exp1001.common.evaluation.base_evaluation import EvaluationBase
import numpy as np


if __name__ == '__main__':

    reward1 = DemoPenalty(weight_matrix=[1,10])
    reward2 = DemoPenalty()
    reward3 = DemoPenalty()
    env1 = Env(
        dt=1,
        size_yudc=[2, 2, 0, 2],
        y_low=[-15, -15],
        y_high=[15, 15],
        u_high=[2, 2],
        u_low=[-2, -2],
        reward_calculator=reward1,
        normalize=False
    )
    env2 = Env(
        dt=1,
        size_yudc=[2, 2, 0, 2],
        y_low=[-15, -15],
        y_high=[15, 15],
        u_high=[2, 2],
        u_low=[-2, -2],
        reward_calculator=reward2,
        normalize=False
    )

    env3 = Env(
        dt=1,
        size_yudc=[2, 2, 0, 2],
        y_low=[-15, -15],
        y_high=[15, 15],
        u_high=[2, 2],
        u_low=[-2, -2],
        reward_calculator=reward3,
        normalize=False
    )
    replay_buffer1 = ReplayBuffer(1000)
    replay_buffer2 = ReplayBuffer(100)

    exploration_noise1 = EGreedy(action_bounds = env1.u_bounds,
                                 epsilon_start=0.5,
                                 epsilon_final=0.4,
                                 epsilon_decay=100000,
                                 )
    exploration_noise1 = No_Exploration()

    exploration_noise2 = GaussianExploration(
        action_bounds=env2.external_u_bounds,
        min_sigma=1.0,
        max_sigma=1.01,
        decay_period=100000)

    controller1 = Td3(
        gpu_id=1,
        num_inputs=env1.observation_size(),
        num_actions=2,
        act_hidden_size=16,
        val_hidden_size=16,
        replay_buffer=replay_buffer1,
        u_bounds=env1.u_bounds,
        exploration=exploration_noise1,
        batch_size=32,
        policy_lr=1e-3,
        value_lr=1e-3,
        noise_std=0.1,
        noise_clip=0.5,
        gamma=0.999,
        policy_update=5,
        soft_tau=1e-3
    )
    controller2 = Td3(
        gpu_id=2,
        num_inputs=env2.observation_size(),
        num_actions=2,
        act_hidden_size=32,
        val_hidden_size=32,
        replay_buffer=replay_buffer2,
        u_bounds=env2.u_bounds,
        exploration=exploration_noise2,
        batch_size=32,
        policy_lr=1e-3,
        value_lr=1e-3,
        noise_std=0.1,
        noise_clip=0.5,
        gamma=0.999,
        policy_update=5,
        soft_tau=1e-3
    )

    controller3 = DemoControl(u_bounds=env2.u_bounds)
    #controller.render_mode = True
    #env.render_mode = True

    exp1 = BaseExp(
        env=env1,
        controller=controller1,
        max_frame = 100000,
        rounds= 10,
        max_step=10,
        eval_rounds=5,
        eval_length=None,
        exp_name="exploration_noise=1.1"
    )

    exp2 = BaseExp(
        env=env2,
        controller=controller2,
        max_frame = 100000,
        rounds= 10,
        max_step=10,
        eval_rounds=5,
        eval_length=None,
        exp_name="exploration_noise=1.01"
    )
    exp3 = BaseExp(
        env=env3,
        controller=controller3,
        max_frame = 100000,
        rounds= 10,
        max_step=10,
        eval_rounds=5,
        eval_length=None,
        exp_name="best"
    )
    #exp.render_mode = True
    #controller2.render_mode = True
    #env2.render_mode = True
    #exp_res1 = exp1.run()
    #exp_res2 = exp2.run()
    #res_list = exp_multi_thread_run([exp1, exp2, exp3])
    res_list = exp_multi_thread_run([exp1, exp2, exp3])
    evaluation = EvaluationBase(res_list=res_list,
                                training_rounds=exp1.rounds,
                                exp_name=["td3_1.1", "td3_1.01", "best"],
                                y_name=["height","concentration"],
                                y_num=2,
                                reward_plt_param={"figsize":(10,5)},
                                eval_plt_param={"figsize":(10,8)}
                                )
    evaluation.draw_rewards()
    evaluation.draw_eval()
    evaluation.error_eval()




