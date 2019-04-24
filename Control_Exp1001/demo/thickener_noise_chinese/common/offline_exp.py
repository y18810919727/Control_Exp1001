# -*- coding:utf8 -*-
import pprint

import numpy as np

from Control_Exp1001.exp.data_package import DataPackage
from Control_Exp1001.demo.thickener.one_round_exp import OneRoundExp


# 实验类：用于调度env和controller进行交互，并每隔一定的训练轮次(rounds)，对模型进行一次评估
class OffLineExp(OneRoundExp):
    def __init__(self, env=None, controller=None,
                 max_step=1000,
                 exp_name=None,
                 act_period=6,
                 train_rounds=100,
                 train_step_in_round=1000,
                 ):
        """

        :param env:
        :param controller:
        :param max_step:
        :param exp_name:
        :param act_period:
        :param train_rounds: 离线训练轮次
        :param train_step_in_round: 每轮离线训练使用的数据量
        """

        super(OffLineExp, self).__init__(env=env, controller=controller, max_step=max_step,
                                         exp_name=exp_name, act_period=act_period)
        self.train_rounds = train_rounds
        self.train_step_in_round = train_step_in_round

    def add_log(self, key, value):
        self.log[key] = value

    def render(self):

        print('************Exp**************')
        # print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print('************Exp**************')
        print()

    def train_controller(self):

        state = self.env.reset()
        self.controller.step_reset()
        self.controller.env=self.env
        for round_it in range(self.train_rounds):
            print('off line round :', round_it)
            for step in range(self.train_step_in_round):

                action = None
                if step % self.act_period == 0:
                    # 控制器计算策略
                    action = np.random.uniform(self.env.u_bounds[:,0],self.env.u_bounds[:,1])
                # 仿真环境进行反馈
                next_state, r, done, _ = self.env.step(action)

                # 训练模型
                if step % self.act_period == 0:
                    # 控制器计算策略
                    self.controller.train(state, action, next_state, r, done)
                state = next_state

                # 记录单步惩罚
                self.log = {}
                self.add_log("step", step)
                if self.render_mode:
                    self.render()
                if done:
                    break
        return

    def run(self):

        self.train_controller()
        state = self.env.reset()
        self.controller.step_reset()
        self.controller.env=self.env
        # 训练eval_cycle个round之后，进行一次模型评估

        y_data = DataPackage(exp_name=self.exp_name, value_name=self.env.y_name)
        u_data = DataPackage(exp_name=self.exp_name, value_name=self.env.u_name)
        d_data = DataPackage(exp_name=self.exp_name, value_name=self.env.d_name)
        c_data = DataPackage(exp_name=self.exp_name, value_name=self.env.c_name)
        u0_grad_data = DataPackage(exp_name=self.exp_name, value_name=['u0 grad'])
        u1_grad_data = DataPackage(exp_name=self.exp_name, value_name=['u1 grad'])
        y0_grad_data = DataPackage(exp_name=self.exp_name, value_name=['y0 grad'])
        y1_grad_data = DataPackage(exp_name=self.exp_name, value_name=['y1 grad'])
        penalty_data = DataPackage(exp_name=self.exp_name, value_name=['cost'])

        y_data.push(self.env.y_star[np.newaxis, :], 'set point')
        y_data.push(self.env.y[np.newaxis, :])

        for step in range(self.max_step):

            if step % self.act_period == 0:
                # 控制器计算策略
                action = self.controller.act(state)
            # 仿真环境进行反馈
            next_state, r, done, _ = self.env.step(action)
            # 训练模型

            if step % self.act_period == 0:
            # 控制器计算策略
                self.controller.train(state, action, next_state, r, done)
            state = next_state

            # 记录单步惩罚
            penalty_data.push(r)
            # 记录目标值
            y_data.push(self.env.y_star[np.newaxis, :], 'set point')
            y_data.push(self.env.y)
            # 记录控制结果
            u_data.push(self.env.u)
            c_data.push(self.env.c)
            d_data.push(self.env.d)
            self.log = {}
            self.add_log("step", step)
            if self.render_mode:
                self.render()
            if done:
                break

        return y_data, u_data, c_data, d_data, penalty_data, u0_grad_data, u1_grad_data, y0_grad_data, y1_grad_data
