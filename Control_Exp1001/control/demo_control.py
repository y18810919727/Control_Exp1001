
# -*- coding:utf8 -*-
from Control_Exp1001.control.base_control import ControlBase

import numpy as np

class DemoControl(ControlBase):

    def __init__(self, u_bounds):
        super(DemoControl,self).__init__(u_bounds)

    def _act(self, state):
        y_star = state[0:2]
        y = state[2:4]
        u = np.clip(y_star - y, self.u_bounds[:,0], self.u_bounds[:, 1])
        return u

    def _train(self, s, u, ns, r, done):
        pass
