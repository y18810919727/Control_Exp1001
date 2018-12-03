import numpy as np
from rewards.base_reward_cal import BaseReward


class DemoReward(BaseReward):

    def cal_reward(self, y_star, y, u, c, d):
        return np.sum((y_star-y)*(y_star-y))
