#!/usr/bin/python
import pprint
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json




from Control_Exp1001.exp.base_exp import BaseExp

class FlotationExp(BaseExp):

    def __init__(self, env=None, controller=None,
                 max_step=1000,
                 exp_name="ILPL"):
        """

        :param env: 仿真环境
        :param controller: 控制模型
        :param max_frame: 最大迭代次数
        :param rounds: 训练回合数
        :param max_step: 每个回合内的最大迭代次数
        :param eval_rounds: 评估轮次
        :param eval_length: 每次评估过程中，迭代次数
        :param exp_name: 实验名称
        """
        super(FlotationExp, self).__init__(
            env=env,
            controller=controller,
            max_frame=max_step,
            rounds=1,
            max_step=max_step,
            eval_rounds=0,
            eval_length=None,
            exp_name=exp_name
        )

    def add_log(self,key, value):
        self.log[key] = value

    def render(self):

        print('************Exp**************')
        #print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print('************Exp**************')
        print()

    def run(self):
        penaltys = []
        eval_penalty_list = []
        # 训练eval_cycle个round之后，进行一次模型评估
        eval_cycle = int(self.rounds/self.eval_rounds)
        frame = 0
        self.env.reset()
        for round_i in range(self.rounds):
            #print(round_i)

            print("Exp :%s, Current round: %i" % ( self.exp_name, round_i))
            state = self.env.reset()
            self.controller.step_reset()
            penalty_sum = 0
            step_in_percent = self.max_step/100.0
            for step in range(self.max_step):
                if step!=0 and step%step_in_percent==0:
                    print(" %d%% End" % (step/step_in_percent))

                # 控制器计算策略
                action = self.controller.act(state)
                # 仿真环境进行反馈
                next_state, r, done, _ = self.env.step(action)
                # 训练模型
                self.controller.train(state, action, next_state, r, done)
                state = next_state
                penalty_sum += r
                frame += 1
                self.log = {}
                self.add_log("frame", frame)
                self.add_log("step", step)
                self.add_log("round", round_i)
                self.add_log("eval_cycle", eval_cycle)
                if self.render_mode:
                    self.render()
                if done:
                    break
                if frame > self.max_frame:
                    break
            penaltys.append(penalty_sum)


            if frame > self.max_frame:
                break

        # 全部训练完，再评估一次
        eval_list = self.controller.plt_list()
        for plt_obj in eval_list:
            plt_obj.plot()

        return penaltys, eval_list, eval_penalty_list
