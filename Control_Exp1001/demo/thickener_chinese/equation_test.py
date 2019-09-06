#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import torch


from torch import nn

net = nn.Sequential(
    nn.Linear(3,3,bias=False),
    nn.Tanh(),
    nn.Linear(3,1,bias=False)
)

print(net)
x = torch.randn((1,3))

def manual(ec,wc2,wc1,x):
    grad_wc1 = 2*ec*(
        wc2.t()*(
                torch.ones(wc1.mm(x.t()).shape) - torch.tanh(
                    wc1.mm(x.t())
                )**2
        )
    ).mm(x)
    grad_wc2 = 2*ec*(torch.tanh(wc1.mm(x.t())).t())
    return grad_wc1, grad_wc2

def test_grad(x):
    out = net(x)
    a = torch.ones(out.shape)
    r = torch.zeros(out.shape)
    td_error = r+0.9*a-out
    loss = td_error*td_error
    loss = loss.mean()
    loss.backward()
    grad1 = (net._modules['0']._parameters['weight']._grad, net._modules['2']._parameters['weight']._grad)
    grad2 = manual(float(td_error.cpu()),
                   net._modules['2']._parameters['weight'].data,
                   net._modules['0']._parameters['weight'].data,
                   x)
    print(grad1, grad2)
test_grad(x)
def test_tanh():
    x = torch.randn((3,2))
    y = torch.tanh(x)
    print(torch.autograd(y.sum(), x))
    print()
