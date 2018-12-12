import numpy as np
import math
import random


class EGreedy:
    def __init__(self, action_bounds, epsilon_start=1, epsilon_final=0.01,
                 epsilon_decay=100000, decay_period=100000):
        self.low = action_bounds[:, 0]
        self.high = action_bounds[:, 1]
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.decay_period = decay_period

    def get_action(self, actions, t=0):
        t = min(self.decay_period, t)
        epsilon = self.epsilon_final+(self.epsilon_start-self.epsilon_final)*math.exp(-1.*t/self.epsilon_decay)
        if random.random() > epsilon:
            return actions
        else:
            np.random.uniform(self.low, self.high)


