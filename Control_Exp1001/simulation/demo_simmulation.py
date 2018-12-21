
# -*- coding:utf8 -*-
import numpy as np
from Control_Exp1001.simulation.base_env import BaseEnv
import random


class DemoSimulation(BaseEnv):

    def __init__(self, dt=1, reward_calculator=None,
                 size_yudc=None,
                 y_low=None, u_low=None,
                 d_low=None, c_low=None,
                 y_high=None, u_high=None,
                 d_high=None, c_high=None,
                 normalize=True
                 ):
        if size_yudc is None:
            size_yudc = [2, 2, 0, 0]

        super(DemoSimulation, self).__init__(dt, reward_calculator, size_yudc,
                                             y_low, u_low,
                                             d_low, c_low,
                                             y_high, u_high,
                                             d_high, c_high, normalize)
        self.y_begin = np.array([0, 0], dtype=float)
        self.u_begin = np.array([0, 0], dtype=float)

    def observation(self):
        return np.concatenate((self.y_star, self.y, self.u, self.c))

    def reset_y(self):
        return self.y_begin + np.array([random.uniform(-2,2),random.uniform(-2,2)],dtype=float)

    def reset_y_star(self):
        return np.array([0,0], dtype=float)

    def reset_u(self):
        return self.u_begin + np.array([random.uniform(-1,1),random.uniform(-1,1)],dtype=float)

    def reset_c(self):
        return np.random.multivariate_normal(np.zeros(2), 1*np.diag(np.ones(2)))

    def f(self, y, u, c, d):
        y = y + u
        y = y + self.c
        c = self.reset_c()
        #c = c + u[0]
        return y, u, c, d



