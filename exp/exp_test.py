from common.action_noise.gaussian_noise import GaussianExploration
from matplotlib import pyplot  as plt
from exp.base_exp import BaseExp
from common.replay.replay_buffer import ReplayBuffer
from common.rewards.demo_reward import DemoReward
from simulation.demo_simmulation import DemoSimulation as Env
from control.td3 import Td3
from control.demo_control import DemoControl
from common.evaluation.base_evaluation import EvaluationBase


if __name__ == '__main__':

    reward = DemoReward()
    env = Env(
        dt=1,
        size_yudc=[2, 2, 0, 2],
        y_low=[-15, -15],
        y_high=[15, 15],
        u_high=[2, 2],
        u_low=[-2, -2],
        reward_calculator=reward
    )

    replay_buffer = ReplayBuffer(1000)

    exploration_noise = GaussianExploration(
        action_bounds=env.external_u_bounds,
        min_sigma=1.0,
        max_sigma=1.1,
        decay_period=100000)

    controller = Td3(
        gpu_id=1,
        num_inputs=env.observation_size(),
        num_actions=2,
        hidden_size=16,
        replay_buffer=replay_buffer,
        u_bounds=env.u_bounds,
        exploration=exploration_noise,
        batch_size=32,
        policy_lr=1e-3,
        value_lr=1e-3,
        noise_std=0.2,
        noise_clip=0.5,
        gamma=0.999,
        policy_update=5,
        soft_tau=1e-3
    )

    env2 = Env(
        dt=1,
        size_yudc=[2, 2, 0, 2],
        y_low=[-15, -15],
        y_high=[15, 15],
        u_high=[2, 2],
        u_low=[-2, -2],
        reward_calculator=reward,
        normalize=False
    )
    controller2 = DemoControl(u_bounds=env2.u_bounds)
    #controller.render_mode = True
    #env.render_mode = True

    exp1 = BaseExp(
        env=env,
        controller=controller,
        max_frame = 100000,
        rounds= 100,
        max_step=300,
        eval_rounds=5,
        eval_length=None
    )

    exp2 = BaseExp(
        env=env2,
        controller=controller2,
        max_frame = 100000,
        rounds= 100,
        max_step=300,
        eval_rounds=5,
        eval_length=None
    )
    #exp.render_mode = True
    controller2.render_mode = True
    env2.render_mode = True
    exp_res1 = exp1.run()
    exp_res2 = exp2.run()
    evaluation = EvaluationBase(res_list=[exp_res1,exp_res2],
                                training_rounds=100,
                                exp_name=["td3", "greedy"],
                                y_name=["height","concentration"],
                                y_num=2,
                                reward_plt_param={"figsize":(10,10)},
                                eval_plt_param={"figsize":(10,8)}
                                )
    evaluation.draw_rewards()
    evaluation.draw_eval()




