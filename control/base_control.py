import pprint
import numpy as np


class ControlBase:
    def __init__(self,
                 u_bounds):

        self.log = {}
        self.buffer = []
        self.buffer_capacity = 0
        self.buffer_position = -1
        # step为当前round下第几步
        self.step = 0
        self.u_bounds = u_bounds
        self.step_reset()
        self.render_mode = False

    def add_log(self,key, value):
        self.log[str(key)] = value

    def step_reset(self):
        self.step = 0
        pass

    def _act(self, state):
        raise NotImplementedError

    def act(self, state):
        self.log = {}
        self.add_log("s", str(state))

        u = self._act(state)

        self.add_log("Type","act")
        self.add_log("u", u)
        if self.render_mode is True:
            self.render()
        self.step += 1
        return u

    def train(self, s, u, ns, r, done):

        self.log = {}

        self._train(s, u, ns, r, done)
        self.add_log("Type","train")

        if self.render_mode is True:
            self.render()

    def _train(self, s, u, ns, r, done):
        raise NotImplementedError

    # 输出日志
    def render(self):
        print('--------Control-------------')
        print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print()
        print('----------------------------')




