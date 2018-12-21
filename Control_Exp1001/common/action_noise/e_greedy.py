
# -*- coding:utf8 -*-
import numpy as np
import math
import random
random.seed()



class EGreedy:
    def __init__(self, action_bounds, epsilon_start=1, epsilon_final=0.01,
                 epsilon_decay=100000):
        """

        :param action_bounds: 动作的上下限
        :param epsilon_start: 起始随机概率
        :param epsilon_final: 最终随机贪婪
        :param epsilon_decay: 折扣因子，因子越小随机概率下降越快，因子越大随机概率下降越慢
        """
        self.low = action_bounds[:, 0]
        self.high = action_bounds[:, 1]
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

    def add_noise(self, actions, t=0):
        # 计算随机概率
        epsilon = self.epsilon_final+(self.epsilon_start-self.epsilon_final)*math.exp(-1.*t/self.epsilon_decay)
        if random.random() > epsilon:
            return actions
        else:
            return np.random.uniform(self.low, self.high)


