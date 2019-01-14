
# -*- coding:utf8 -*-
import os
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
from Control_Exp1001.simulation import utils


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
        self.u_bounds = None
        self.y_shape = 0
        self.u_shape = 0


        # 控制项的惩罚权重
        if S is None :
            S = np.ones(self.u_shape)
        S = np.diag(S)
        self.S = S


        # 目标项的惩罚权重
        if weight_matrix is None:
            weight_matrix = np.ones(self.y_shape)
        weight_matrix = np.diag(weight_matrix)
        self.weight_matrix = weight_matrix



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




