# -*- coding:utf8 -*-
import numpy as np
class GaussianExploration:
    def __init__(self,action_bounds, min_sigma = 1.0, max_sigma=1.0,decay_period=1000000):
        """

        对action添加高斯噪音，适用于将action归一到(-1,1)区间的的仿真模型
        :param action_bounds: 动作上下限
        :param min_sigma: 噪声放缩倍率下限
        :param max_sigma: 噪声放缩倍率下限
        :param decay_period: 折扣因子，因子越大

        """
        self.low = action_bounds[:, 0]
        self.high= action_bounds[:, 1]
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def add_noise(self, actions, t=0):
        sigma = self.max_sigma - (self.max_sigma-self.min_sigma) * min(1.0,t/self.decay_period)
        noise = np.random.normal(size=len(actions))*sigma
        noise = np.mat(noise).dot(np.diag(self.high-self.low))
        noise = np.squeeze(np.array(noise),axis=0)
        actions = actions + noise
        actions = np.clip(actions, self.low, self.high)
        return actions
