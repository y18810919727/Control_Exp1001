#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
import os
import torch
import json

class Actor(torch.nn.Module):
    def __init__(self,dim_in,dim_out,device,Va=None,La=None,dim_hidden=4,):
        super(Actor,self).__init__()

        self.Va=torch.nn.Linear(dim_in,dim_hidden,bias=False)
        if Va is not None:
            self.Va.weight.data=torch.FloatTensor(Va)
            self.Va.weight.requires_grad = False

        self.Wa = torch.nn.Linear(dim_hidden,dim_out)

        self.La = torch.nn.Linear(dim_out,dim_out,bias=False)
        self.La.weight.requires_grad = False

        if La is None:
            self.La.weight.data=torch.FloatTensor(np.diag(np.ones(dim_out)))
        else:
            La = torch.FloatTensor(La)
            self.La.weight.data = torch.FloatTensor(La.inverse())

        self.to(device)

    def forward(self, x):
        y = self.Va(x)
        y = torch.tanh(y)
        y = self.Wa(y)
        y = self.La(y)
        return y

