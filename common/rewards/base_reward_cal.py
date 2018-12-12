import os
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
from simulation import utils


class BaseReward:
    def __init__(self):
        self.state_buffer = []
        self.position = -1
        self.capacity = 100

    def push(self, x):
        if len(self.state_buffer) < self.capacity:
            self.state_buffer.append(None)
        self.position = (self.position+1) % self.capacity
        self.state_buffer[self.position] = x

    def get_last_state(self):
        return self.state_buffer[self.position]

    def cal_reward(self,y_star, y, u, c, d):
        raise NotImplementedError

    def cal(self, y_star, y, u, c, d):
        reward = self.cal_reward(y_star, y, u, c, d)
        self.push((y_star, y, u, c, d))
        return reward




