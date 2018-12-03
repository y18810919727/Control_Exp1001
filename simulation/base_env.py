import pprint
import os
import pandas as pd

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
from rewards.demo_reward import DemoReward
from simulation import utils


class BaseEnv():
    """
    Superclass for all simulations.
    """
    def __init__(self, dt=1, reward_calculator=None,
                 size_yudc=None,
                 y_low=None, u_low=None,
                 d_low=None, c_low=None,
                 y_high=None, u_high=None,
                 d_high=None, c_high=None
                 ):
        # time step
        self.dt = dt
        # goal of y
        self.y_star = None
        # y --- core indices
        self.y = None

        # control input
        self.u = None

        # measurable parameters
        self.c = None

        # unmeasurable parameters
        self.d = None
        self.np_random = None
        self.reward = 0
        self.done = False
        if size_yudc is None:
            raise Exception('No size_yudc!')

        self.size_yudc = size_yudc

        # 定义参量边界
        self.y_bounds = None
        self.u_bounds = None
        self.d_bounds = None
        self.c_bounds = None

        # set bounds for yudc
        self.set_y(self.size_yudc[0], y_low, y_high)
        self.set_u(self.size_yudc[1], u_low, u_high)
        self.set_d(self.size_yudc[2], d_low, d_high)
        self.set_c(self.size_yudc[3], c_low, c_high)

        # Each env has a reward calculator, the default calculator is DemoReward.
        if reward_calculator is None:
            reward_calculator = DemoReward()
        self.reward_calculator = reward_calculator

        # store log
        self.log = []

    @staticmethod
    def set_bound(size, low, high, kind='Unkown'):
        if low is None:
            low = np.ones(size)*np.inf*-1
        if high is None:
            high = np.ones(size)*np.inf*1
        if not utils.is_nparray(low) or not utils.is_nparray(high):
            raise TypeError('Bounds should be numpy.ndarray!')
        if low.shape[0] != size or high.shape[0] != size:
            raise ValueError('Shape of bounds is not match to dim %s' % kind )
        return np.concatenate((low.reshape(size, 1), high.reshape(size, 1)), axis=1)

    # 定义目标变量的边界，下同 output
    def set_y(self,size, y_low = None, y_high = None):
        self.y_bounds = self.set_bound(size, y_low, y_high, 'Indices')

    def set_u(self, size, u_low=None, u_high=None):
        self.u_bounds = self.set_bound(size, u_low, u_high, 'Control input')

    def set_d(self,size,d_low = None, d_high=None):
        self.d_bounds = self.set_bound(size, d_low, d_high, 'Unmeasurable parameters')

    def set_c(self, size, c_low=None, c_high=None):
        self.c_bounds = self.set_bound(size, c_low, c_high, 'Measurable parameters')

    def set_y_star(self, y_star):
        if y_star.shape != self.y_bounds[:,0]:
            raise ValueError('Shape of y* is not match to  y')
        self.y_star = y_star

    # clip x in the range of bounds
    def bound_detect(self, x, bounds, item_name='Unknow'):
        if x.shape[0] == 0:
            return 0, 'Normal', x
        low = bounds[:, 0]
        high = bounds[:, 1]
        res = np.clip(x, low, high)
        if (res == low).sum() >= 1:
            self.log.append("Too low for %s" % item_name)
            return -1, 'Too low', res
        if (res == high).sum() >= 1:
            self.log.append("Too large for %s" % item_name)
            return 1, 'Too high', res
        return 0, 'Normal', res

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ----------------------------------------------------------
    # Need to be Implement for reset
    def reset_y(self):
        raise NotImplementedError('Please implement reset_y')

    def reset_u(self):
        raise NotImplementedError('Please implement reset_u')

    def reset_y_star(self):
        raise NotImplementedError('Please implement reset_y_star')

    def reset_c(self):
        return np.array([])

    def reset_d(self):
        return np.array([])
    # ----------------------------------------------------------

    def _reset_y(self):
        y = self.reset_y()
        bound_res = self.bound_detect(y, self.y_bounds, item_name='reset y')
        y = bound_res[2]
        if bound_res[0] is not 0:
            raise ValueError('Reset for y is %s' % bound_res[1])
        if len(y) != self.size_yudc[0]:
            raise ValueError('Shape for y is wrong')

        self.y = y

    def _reset_y_star(self):
        y_star = self.reset_y_star()
        bound_res = self.bound_detect(y_star, self.y_bounds, item_name='reset y*')
        y_star = bound_res[2]
        if bound_res[0] is not 0:
            raise ValueError('Reset for y_star is %s' % bound_res[1])
        if len(y_star) != self.size_yudc[0]:
            raise ValueError('Shape for y_star is wrong')
        self.y_star = y_star

    def _reset_u(self):
        u = self.reset_u()

        if len(u) != self.size_yudc[1]:
            raise ValueError('Shape for u is wrong' )

        bound_res = self.bound_detect(u, self.u_bounds, item_name='reset u')
        u= bound_res[2]
        if bound_res[0] is not 0:
            raise ValueError('Reset for u is %s' % bound_res[1])
        self.u = u

    def _reset_d(self):
        d = self.reset_d()
        if len(d) != self.size_yudc[2]:
            raise ValueError('Shape for d is wrong')
        bound_res = self.bound_detect(d, self.d_bounds, item_name='reset d')
        d = bound_res[2]
        if bound_res[0] is not 0:
            raise ValueError('Reset for d is %s' % bound_res[1])
        self.d = d

    def _reset_c(self):
        c = self.reset_c()
        if len(c)!=self.size_yudc[3]:
            raise ValueError('Shape for c is wrong')
        bound_res = self.bound_detect(c, self.c_bounds, item_name='reset c')
        c = bound_res[2]
        if bound_res[0] is not 0:
            raise ValueError('Reset for c is %s' % bound_res[1])

        self.c = c

    def _reset(self):
        pass

    def reset(self):
        self._reset()
        self._reset_y()
        self._reset_y_star()
        self._reset_u()
        self._reset_d()
        self._reset_c()

    def solve_u(self, u):
        bound_res_u = self.bound_detect(u, self.u_bounds,item_name='simulation u')
        if bound_res_u[0]!=0:
            self.done = True
        return bound_res_u

    def observation(self):
        """
        Could be implemented
        :return:
        """
        return np.concatenate((self.y_star, self.y, self.c))

    def render(self):

        dic = {
            "y_star": self.y_star,
            "y": self.y,
            "u": self.u,
            "d": self.d,
            "c": self.c,
            "reward": self.reward,
            "done": self.done,
            "log": self.log
        }
        pprint.pprint(dic)
        print('----------------------------')

        return None

    def step(self, new_u):
        # clean log
        self.log = []
        self.done = False
        self.reward = 0

        # clip u in u_bounds
        self.u = self.solve_u(new_u)[2]

        # calculate new y u d c ,and whether the env is terminal
        self.done = self._step()

        # calculate the reward according to reward calculator
        self.reward = self.reward_calculator.cal(self.y_star, self.y, self.u, self.c, self.d)

        return self.observation(), self.reward, self.done, None

    def _step(self):

        done = False
        for _ in range(self.dt):
            self.y, self.u, self.c, self.d = self.f(self.y, self.u, self.c, self.d)

            bound_res_y = self.bound_detect(self.y, self.y_bounds,item_name='simulation y')
            self.y = bound_res_y[2]

            bound_res_c = self.bound_detect(self.c, self.c_bounds,item_name='simulation c')
            self.c = bound_res_c[2]

            bound_res_d = self.bound_detect(self.d, self.d_bounds,item_name='simulation d')
            self.d = bound_res_d[2]

            if bound_res_y[0] is not 0:
                #print('Indices are %s' % bound_res_y[1])
                done = True
                break

        return done

    def f(self, y, u, c, d):
        """
        Implement the simulation process in one step
        :return: new y,u,c,d
        """
        return (y,u,c,d)
        #raise NotImplementedError







