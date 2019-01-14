#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json
import torch

class Critic(torch.nn.Module):
    def __init__(self,dim_in,dim_out,device,Vc=None,Lc=None,dim_hidden=4,):
        super(Critic,self).__init__()


        self.Vc=torch.nn.Linear(dim_in,dim_hidden,bias=False)
        if Vc is not None:
            self.Vc.weight.data=torch.FloatTensor(Vc)
            self.Vc.weight.requires_grad = False

        self.Wc = torch.nn.Linear(dim_hidden,dim_out)

        self.Lc = torch.nn.Linear(dim_out,dim_out,bias=False)
        self.Lc.weight.requires_grad = False

        if Lc is None:
            self.Lc.weight.data=torch.FloatTensor(np.diag(np.ones()))
        else:
            Lc = torch.FloatTensor(Lc)
            self.Lc.weight.data=torch.FloatTensor(Lc.inverse())

        self.to(device)

    def forward(self, x):
        y = self.Vc(x)
        y = torch.tanh(y)
        y = self.Wc(y)
        y = self.Lc(y)
        return y
