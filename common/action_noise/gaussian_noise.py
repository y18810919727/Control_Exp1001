import numpy as np
class GaussianExploration:
    def __init__(self,action_bounds, min_sigma = 1.0, max_sigma=1.0,decay_period=1000000):
        self.low = action_bounds[:, 0]
        self.high= action_bounds[:, 1]
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def add_noise(self, actions, t=0):
        sigma = self.max_sigma - (self.max_sigma-self.min_sigma) * min(1.0,t/self.decay_period)
        actions = actions + np.random.normal(size=len(actions))*sigma
        actions = np.clip(actions, self.low, self.high)
        return actions
