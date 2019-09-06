# -*- coding:utf8 -*-
import pprint

import numpy as np

from Control_Exp1001.demo.thickener_chinese.common.data_package import DataPackage
import matplotlib
#matplotlib.style.use('ggplot')
from sklearn.metrics import mean_absolute_error as mae

import pickle

# 实验类：用于调度env和controller进行交互，并每隔一定的训练轮次(rounds)，对模型进行一次评估
class OneRoundExp:
    def __init__(self, env=None, controller=None,
                 max_step=1000,
                 exp_name=None,
                 act_period=6,
                 ):
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
        self.act_period = act_period


    def add_log(self, key, value):
        self.log[key] = value

    def render(self):

        print('************Exp**************')
        # print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print('************Exp**************')
        print()



    def run(self):
        import os
        if os.path.exists(
            os.path.join('..','result',self.exp_name)
        ):
            with open(
                os.path.join('..','result',self.exp_name)
            ,'rb') as f:
                obj = pickle.load(f)
            res = obj
            return res


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
        penalty_data = DataPackage(exp_name=self.exp_name, value_name=['效用值'])

        y_data.push(self.env.y_star[np.newaxis, :], '设定值')
        y_data.push(self.env.y[np.newaxis, :])
        controller_step = int(self.max_step/self.act_period)

        for step in range(controller_step):
            action = self.controller.act(state)
            first_r = None
            final_next_state = None
            done = False
            for _ in range(self.act_period):
                next_state, r, done, _ = self.env.step(action)
                penalty_data.push(r)
                # 记录目标值
                y_data.push(self.env.y_star[np.newaxis, :], '设定值')
                y_data.push(self.env.y)
                # 记录控制结果
                u_data.push(self.env.u)
                c_data.push(self.env.c)
                d_data.push(self.env.d)
                # 效用函数值为第一时刻的xk, uk
                if first_r is None:
                    first_r = r
                final_next_state = next_state # 取最后一个next_state

            self.controller.train(state, action, final_next_state, first_r, done)
            state = final_next_state
        y_array = np.array(y_data)
        other_info = {}
        other_info['time_used'] =(self.controller.time_used, self.controller.train_time_used,
                                  self.controller.act_time_used)
        other_info['exp_name']=self.exp_name
        res = y_data, u_data, c_data, d_data, penalty_data, u0_grad_data, u1_grad_data, y0_grad_data, y1_grad_data, \
               other_info

        with open(
                os.path.join('..','result',self.exp_name)
                ,'wb') as f:
            pickle.dump(res, f)
        return res
