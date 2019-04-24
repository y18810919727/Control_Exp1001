
# -*- coding:utf8 -*-
import numpy as np
import random
# 经验回放池普通版
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.item_id = 0
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, self.item_id)
        self.item_id += 1
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, step=None):
        if batch_size>=self.__len__():
            batch = self.buffer
        else:
            batch = sorted(random.sample(self.buffer, batch_size), key=lambda x : x[5])
        state, action, reward, next_state, done, item_id = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

