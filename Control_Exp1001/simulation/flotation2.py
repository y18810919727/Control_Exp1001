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
    def __init__(self, dt=1, penalty_calculator=None,
                 size_yudc=None, u_low=np.array([1,10], dtype=float),
                 u_high=np.array([3,30],dtype=float), normalize=False,
                 time_length=1,
                 one_step_length=0.001,
                 y_name=None,
                 y_star=None,
                 c_name=None,
                 ):
        if size_yudc is None:
            size_yudc = [2, 2, 0, 4]
        if y_name is None:
            y_name = ["Lcg", "Ltg"]

        if c_name is None:
            c_name = ["Mp1", "Mp2", "Me1", "Me2"]

        super(Flotation, self).__init__(
            dt = dt,
            penalty_calculator=penalty_calculator,
            size_yudc=size_yudc,
            u_low=u_low,
            u_high=u_high,
            normalize=normalize,
            time_length=time_length,
            one_step_length=one_step_length,
            y_name=y_name,
            c_name=c_name
        )
        if y_star is None:
            y_star = np.array([17.34, 0.75], dtype=float)
        self.y_star = y_star
        self.ke = np.array([65.6,316], dtype=float)  # Drainage rate
        self.kp = np.array([17.9,0.04], dtype=float)  # flotation rate
        self.gcp = np.array([0.417, 0.0034], dtype=float)  #
        self.ga = 0.0234  # feed mineral grade
        self.A = 53.2 # sectional area
        self.H = 3.2  # total height
        self.Lcu = 42.1  #
        self.qT = 9.3  # tail flow
        #self.qc = 20  #
        self.qc = 7.392  #
        self.Xa = np.array([0,0],dtype=float)  # mineral grade
        #self.Xa[0] = 0.4
        self.Xa[0] = 0.1549
        self.Xa[1] = (self.gcp[0]-self.ga)/(self.ga-self.gcp[1]) * self.Xa[0]

    def reset_u(self):
        return np.array([1,10],dtype=float)

    def reset_y(self):
        Mp = self.c[:2]  # pulp mass
        Me = self.c[2:]  # froth mass
        lcg = np.sum(self.gcp * Me) * self.Lcu / (np.sum(Me))  # general concentrate grade
        ltg = np.sum(self.gcp * Mp) * self.Lcu / (np.sum(Mp))  # general tail grade
        #ny = np.array([lcg, ltg], dtype=float)
        ny = np.hstack([lcg, ltg])
        return ny

    def reset_c(self):

        c = np.array([28.12, 780, 7.36, 0.098], dtype=float)
        #c = np.array([16.8, 1123, 4.56, 0.2],dtype=float)
        return c

    def reset_y_star(self):
        #return np.array([17.34, 0.75],dtype=float)
        #return np.array([16.8, 1123, 4.3, 0.12], dtype=float)
        return self.y_star

    def observation(self):
        return np.hstack([self.y_star, self.y, self.u])

    def f(self, y, u, c, d):

        hp, qa = tuple(u.tolist())

        Mp = c[:2]
        Me = c[2:]

        self.qT = 0.9679*(qa - self.qc)

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

    @classmethod
    def flotation_test(cls, init_para=None):
        if init_para is None:
            init_para = {}
        simulation_test(Flotation, mode="const",init_para=init_para,
                        const_u=[[1, 17], [1.5, 17], [2, 10], [2, 20], [2.5, 17]], seprate_num=3,
                        test_step=100, eval_plt_param={"figsize": (15, 10)})
        simulation_test(Flotation, mode="random",init_para=init_para,test_step=100, eval_plt_param={"figsize": (15, 10)})

        simulation_test(Flotation, mode="uniform",init_para=init_para, seprate_num=3, test_step=100, eval_plt_param={"figsize": (15, 10)})

if __name__ == '__main__':
    '''
    
    simulation_test(Flotation, mode="const",
                    const_u=[[1, 17], [1.5, 17], [2, 3], [2, 20], [2.5, 17]] , seprate_num=3,
                    test_step=200, eval_plt_param={"figsize": (15, 10)})
    '''
    # simulation_test(Flotation, mode="random", test_step=200, eval_plt_param={"figsize": (15, 10)})
    #
    # simulation_test(Flotation, mode="uniform", seprate_num=3, test_step=200, eval_plt_param={"figsize": (15, 10)})
    #
    # simulation_test(Flotation, mode="const",
    #                 const_u=[[1, 17], [1.5, 10], [1.5, 17], [1.5, 20],[1.5, 30], [2.5, 17], [3, 17]] , seprate_num=3,
    #                 test_step=200, eval_plt_param={"figsize": (15, 10)})
    #
    # simulation_test(Flotation, mode="const",
    #                 const_u=[[2.232,19.85]] , seprate_num=3,
    #                 test_step=200, eval_plt_param={"figsize": (15, 10)})
    simulation_test(Flotation, mode="const",
                    const_u=[[2.28, 19.5],[2.28, 20.5],[2.30, 19.5],[2.30, 20.5]],
                    test_step=200, eval_plt_param={"figsize": (15, 10)})

