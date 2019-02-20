# -*- coding:utf8 -*-
import pprint

import numpy as np

from Control_Exp1001.exp.data_package import DataPackage


# 实验类：用于调度env和controller进行交互，并每隔一定的训练轮次(rounds)，对模型进行一次评估
class OneRoundExp:
    def __init__(self, env=None, controller=None,
                 max_step=1000,
                 exp_name=None):
        """

        :param env:
        :param controller:
        :param max_step: 仿真迭代次数
        :param exp_name:
        """

        self.env = env
        self.controller = controller
        # 每个round的迭代次数
        self.max_step = max_step
        # 总迭代次数上限
        self.render_mode = False
        self.log = {}
        if exp_name is None:
            exp_name = "None"
        self.exp_name = exp_name

    def add_log(self, key, value):
        self.log[key] = value

    def render(self):

        print('************Exp**************')
        # print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print('************Exp**************')
        print()

    def run(self):

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

            # 控制器计算策略
            action = self.controller.act(state)
            # 仿真环境进行反馈
            next_state, r, done, _ = self.env.step(action)
            # 训练模型
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
            u0_grad_data.push(float(self.controller.u_grad[-1][0]))
            u1_grad_data.push(float(self.controller.u_grad[-1][1]))
            y0_grad_data.push(float(self.controller.y_grad[-1][0]))
            y1_grad_data.push(float(self.controller.y_grad[-1][1]))
            self.log = {}
            self.add_log("step", step)
            if self.render_mode:
                self.render()
            if done:
                break

        return y_data, u_data, c_data, d_data, penalty_data, u0_grad_data, u1_grad_data, y0_grad_data, y1_grad_data
