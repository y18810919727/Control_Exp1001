#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import numpy as np
import math
import Control_Exp1001 as CE
import os
import json

class Model(torch.nn.Module):
    def __init__(self,dim_in,dim_out,device,Vm=None,Lm=None,dim_hidden=4,):
        super(Model,self).__init__()

        self.Vm=torch.nn.Linear(dim_in,dim_hidden,bias=False)
        if Vm is not None:
            self.Vm.weight.data=torch.FloatTensor(Vm)
            self.Vm.weight.requires_grad = False

        self.Wm = torch.nn.Linear(dim_hidden,dim_out)

        self.Lm = torch.nn.Linear(dim_out,dim_out,bias=False)
        self.Lm.weight.requires_grad = False

        if Lm is None:
            self.Lm.weight.data=torch.FloatTensor(np.diag(np.ones()))
        else:
            Lm = torch.FloatTensor(Lm)
            self.Lm.weight.data=torch.FloatTensor(Lm.inverse())

        self.to(device)

    def forward(self, x):
        y = self.Vm(x)
        y = torch.tanh(y)
        y = self.Wm(y)
        y = self.Lm(y)
        return y
