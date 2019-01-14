#!/usr/bin/python
# -*- coding:utf8 -*-
from scipy.integrate import quad
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json
from Control_Exp1001.simulation.flotation import Flotation

from Control_Exp1001.common.penaltys.base_penalty_cal import BasePenalty
class IntegralPenalty(BasePenalty):

    def cal_penalty(self, y_star, y, u, c, d):
        weight_matrix = self.weight_matrix



        y_size = np.prod(y_star.shape)
        diff_y = (y_star-y).reshape(1, y_size)


        """
        diff is a row vector
        res = diff * W * diff.T
        """
        penalty_y = diff_y.dot(weight_matrix).dot(diff_y.T)

        penalty_u_quad = quad(
            lambda x,u_bounds=self.u_bounds,u=u:self.f(x, u_bounds,u), 0, 1)

        penalty_u = penalty_u_quad[0]


        penalty = (penalty_y.data[0, 0] + penalty_u)
        return penalty


    def f(self, rate, u_bounds, u):

        u_mid = np.mean(self.u_bounds, axis=1)
        x = u_mid + rate*(u - u_mid)
        x = x.reshape(-1,1)
        u_mid = u_mid.reshape(-1,1)
        u_all = x - u_mid

        U = np.diag((u_bounds[:,1]-u_bounds[:,0])/2)
        S = self.S
        R_s = np.arctanh(np.linalg.inv(U).dot(x-u_mid)).T.dot(U.dot(S))
        R_s = R_s.reshape(1,-1)
        u_all = u_all.reshape(-1, 1)
        res = R_s.dot(u_all)
        return res







if __name__ == '__main__':
    penalty_cal = IntegralPenalty(weight_matrix=[1,1], S=[0.5,0.5])
    env = Flotation(penalty_calculator=penalty_cal)

    print(env.step(np.array([1,30])))
    print(env.step(np.array([2,20])))
    print(env.step(np.array([1.5,10])))


