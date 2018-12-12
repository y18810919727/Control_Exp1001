from control.base_control import ControlBase
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np


class ACBase(ControlBase):
    def __init__(self, gpu_id=0,
                 replay_buffer = None,
                 u_bounds = None,
                 exploration = None,
                 ):
        super(ACBase, self).__init__(u_bounds)
        self.device = None
        self.cude_device(gpu_id)
        self.replay_buffer = replay_buffer
        self.exploration = exploration
        self._train_mode = True


    @property
    def train_mode(self):
        return self._train_mode

    @train_mode.setter
    def train_mode(self,new_state):
        print('train_mode is changed to %s!' % "True" if new_state else "False")
        self._train_mode = new_state

    def cude_device(self, cuda_id):
        use_cuda = torch.cuda.is_available()
        cuda = 'cuda:'+str(cuda_id)
        self.device = torch.device(cuda if use_cuda else "cpu")

    def _train(self, s, u, ns, r, done):
        raise NotImplementedError

    def policy_act(self, state):
        raise NotImplementedError

    def _act(self, state):

        # 计算增加噪音之前的的策略
        action_before_noise = self.policy_act(state)
        if self.train_mode:

            # 训练过程中加入噪音，用于探索
            action_after_noise = self.exploration.add_noise(action_before_noise, self.step)
        else:
            action_after_noise = action_before_noise
        self.add_log("Action (After adding noise)", action_after_noise)
        self.add_log("Action (Before adding noise)", action_before_noise)
        return action_after_noise






    # TODO save models list
    def save_model(self,model_list = None, name_list = None):
        if model_list is None:
            pass

    # TODO load models list
    def load_model(self):
        pass


