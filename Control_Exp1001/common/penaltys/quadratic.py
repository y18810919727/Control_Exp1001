
# -*- coding:utf8 -*-
import numpy as np

from Control_Exp1001.common.penaltys.base_penalty_cal import BasePenalty


# 最简单的奖赏计算
class Quadratic(BasePenalty):

    def cal_penalty(self, y_star, y, u, c, d):
        weight_matrix = self.weight_matrix
        y_size = np.prod(y_star.shape)

        u_mid = np.mean(self.u_bounds, axis=1)

        tmp = (y_star-y).reshape(1, y_size)
        det_u = (u-u_mid).reshape(1,-1)

        """
        a is a row vector
        res = a * W * a.T + u * S * u.T
        """
        penalty_u = float(det_u.dot(self.S).dot(det_u.T))
        #penalty_u = penalty_u*penalty_u*penalty_u*100000
        res = float(tmp.dot(weight_matrix).dot(tmp.T)) + penalty_u
        return res
