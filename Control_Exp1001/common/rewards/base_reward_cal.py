
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
class BaseReward:
    def __init__(self, weight_matrix=None):

        self.state_buffer = [] #存储历史片段
        self.position = -1
        self.capacity = 100
        self.weight_matrix = weight_matrix


    def push(self, x):
        if len(self.state_buffer) < self.capacity:
            self.state_buffer.append(None)
        self.position = (self.position+1) % self.capacity
        self.state_buffer[self.position] = x

    def get_last_state(self):
        return self.state_buffer[self.position]

    # 必须重写
    def cal_reward(self,y_star, y, u, c, d, weight):
        raise NotImplementedError

    def cal(self, y_star, y, u, c, d):
        weight_matrix = self.weight_matrix
        if weight_matrix is None:
            weight_matrix = np.ones(y.shape)
        weight_matrix = np.diag(weight_matrix)
        reward = self.cal_reward(y_star, y, u, c, d, weight_matrix)
        self.push((y_star, y, u, c, d))
        return reward




