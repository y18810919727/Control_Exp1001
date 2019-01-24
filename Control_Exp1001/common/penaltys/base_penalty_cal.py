
# -*- coding:utf8 -*-
import numpy as np


# 计算奖赏，需要继承
class BasePenalty:
    def __init__(self, weight_matrix=None, S=None):
        """
        Norm of weight_matrix should be bigger than S
        :param weight_matrix:
        :param S:
        """

        self.state_buffer = [] #存储历史片段
        self.position = -1
        self.capacity = 100
        self.weight_matrix = weight_matrix
        self.S = S
        self.u_bounds = None
        self.y_shape = 0
        self.u_shape = 0

    def change_list2diag(self, size, W=None):
        if W is None:
            W = np.diag(np.ones(size))
        else:
            W = np.array(W)
            if W.shape[0] != size:
                raise ValueError("size mismatch")
            W = np.diag(W)
        return W

    def create(self):

        self.weight_matrix = self.change_list2diag(self.y_shape, self.weight_matrix)
        self.S = self.change_list2diag(self.u_shape, self.S)

    def push(self, x):
        if len(self.state_buffer) < self.capacity:
            self.state_buffer.append(None)
        self.position = (self.position+1) % self.capacity
        self.state_buffer[self.position] = x

    def get_last_state(self):
        return self.state_buffer[self.position]

    # 必须重写
    def cal_penalty(self,y_star, y, u, c, d):
        raise NotImplementedError

    def cal(self, y_star, y, u, c, d):
        penalty = self.cal_penalty(y_star, y, u, c, d)
        self.push((y_star, y, u, c, d))
        return penalty




