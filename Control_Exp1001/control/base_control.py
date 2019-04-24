
# -*- coding:utf8 -*-
import pprint

import numpy as np
import time


# 控制基础类
class ControlBase:
    def __init__(self,
                 u_bounds):

        self.log = {}
        self.buffer = []
        self.buffer_capacity = 0
        self.buffer_position = -1
        # step为当前round下第已经训练了几步
        self.step = 0
        self.u_bounds = u_bounds
        self.step_reset()
        self.render_mode = False
        self.time_used = 0
        self.act_time_used = 0
        self.train_time_used= 0

    def add_log(self,key, value):
        self.log[str(key)] = value

    def step_reset(self):
        self.step = 0
        pass

    # 必须实现，
    def _act(self, state):
        raise NotImplementedError

    def act(self, state):
        self.log = {}
        self.add_log("s", str(state))

        time_begin = time.time()
        u = self._act(state)
        time_end = time.time()
        self.time_used += time_end-time_begin
        self.act_time_used+=(time_end-time_begin)

        self.add_log("Type","act")
        self.add_log("u", u)
        if self.render_mode is True:
            self.render()
        self.step += 1
        return u

    def train(self, s, u, ns, r, done):

        self.log = {}

        time_begin = time.time()
        self._train(s, u, ns, r, done)
        time_end = time.time()
        self.time_used += time_end - time_begin
        self.train_time_used+=(time_end-time_begin)
        self.add_log("Type","train")


        if self.render_mode is True:
            self.render()

    def _train(self, s, u, ns, r, done):
        raise NotImplementedError

    # 输出日志
    def render(self):
        print('--------Control-------------')
        print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print()
        print('----------------------------')




