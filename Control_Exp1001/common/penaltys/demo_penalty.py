
# -*- coding:utf8 -*-
import numpy as np

from Control_Exp1001.common.penaltys.base_penalty_cal import BasePenalty


# 最简单的奖赏计算
class DemoPenalty(BasePenalty):

    def cal_penalty(self, y_star, y, u, c, d):
        weight_matrix = self.weight_matrix
        y_size = np.prod(y_star.shape)

        tmp = (y_star-y).reshape(1, y_size)

        """
        a is a row vector
        res = a * W * a.T
        """
        res = float(tmp.dot(weight_matrix).dot(tmp.T))
        return res
