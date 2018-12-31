#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
from Control_Exp1001.simulation.base_env import BaseEnv
from Control_Exp1001.simulation.simulation_test import simulation_test
import Control_Exp1001.demo.exp_test
import matplotlib.pyplot as plt

class Flotation(BaseEnv):
    def __init__(self, dt=1, reward_calculator=None,
                 size_yudc=None, u_low=np.array([1,3], dtype=float),
                 u_high=np.array([3,30],dtype=float), normalize=False,
                 time_length=1,
                 one_step_length=0.001,
                 y_name=None
                 ):
        if size_yudc is None:
            size_yudc = [2, 2, 0, 4]
        if y_name is None:
            y_name = ["Lcg", "Ltg"]
        super(Flotation, self).__init__(
            dt = dt,
            reward_calculator=reward_calculator,
            size_yudc=size_yudc,
            u_low=u_low,
            u_high=u_high,
            normalize=normalize,
            time_length=time_length,
            one_step_length=one_step_length,
            y_name=y_name
        )
        self.ke = np.array([65.6,316], dtype=float)
        self.kp = np.array([17.9,0.04], dtype=float)
        self.gcp = np.array([0.417, 0.0034], dtype=float)
        self.ga = 0.0234
        self.A = 53.2
        self.H = 3.2
        self.Lcu = 42
        self.qT = 9.3
        self.qc = 20
        self.Xa = np.array([0,0],dtype=float)
        self.Xa[0] = 0.4
        self.Xa[1] = (self.gcp[0]-self.ga)/(self.ga-self.gcp[1]) * self.Xa[0]

    def reset_u(self):
        return np.array([1,10],dtype=float)

    def reset_y(self):
        Mp = self.c[:2]
        Me = self.c[2:]
        lcg = np.sum(self.gcp * Me) * self.Lcu / (np.sum(Me))
        ltg = np.sum(self.gcp * Mp) * self.Lcu / (np.sum(Mp))
        #ny = np.array([lcg, ltg], dtype=float)
        ny = np.hstack([lcg, ltg])
        return ny

    def reset_c(self):

        #c = np.array([16.8, 1123, 8.3, 0.2],dtype=float)
        # 变成[1.5, 17] 情况下的稳态
        c = np.array([28.12, 780, 7.36, 0.098], dtype=float)
        #c = np.array([16.8, 1123, 4.56, 0.2],dtype=float)
        return c

    def reset_y_star(self):
        #return np.array([17.34, 0.75],dtype=float)
        #return np.array([16.8, 1123, 4.3, 0.12], dtype=float)
        return np.array([17.20, 0.60], dtype=float)

    def f(self, y, u, c, d):

        hp, qa = tuple(u.tolist())

        Mp = c[:2]
        Me = c[2:]

        for _ in np.arange(0, self.time_length, self.one_step_length):

            dmp = -(self.kp + self.qT/(self.A*hp))*Mp + self.ke*Me + qa*self.Xa
            dme = -(self.ke + self.qc/(self.A*(self.H - hp)))*Me + self.kp*Mp
            Mp = Mp + self.one_step_length * dmp
            Me = Me + self.one_step_length * dme


        lcg = np.sum(self.gcp * Me)*self.Lcu/(np.sum(Me))
        ltg = np.sum(self.gcp * Mp)*self.Lcu/(np.sum(Mp))
        nc = np.hstack([Mp, Me])
        ny = np.array([lcg, ltg], dtype=float)

        return ny, u, nc, d

if __name__ == '__main__':
    simulation_test(Flotation, mode="const",
                    const_u=[[1, 17], [1.5, 17], [2, 3], [2, 20], [2.5, 17]] , seprate_num=3,
                    test_step=100, eval_plt_param={"figsize": (15, 10)})
    simulation_test(Flotation, mode="random", test_step=100, eval_plt_param={"figsize": (15, 10)})

    simulation_test(Flotation, mode="uniform", seprate_num=3, test_step=100, eval_plt_param={"figsize": (15, 10)})