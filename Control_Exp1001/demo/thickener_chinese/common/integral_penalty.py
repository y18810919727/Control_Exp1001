#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
import time
import torch
from torch.autograd import Function
from scipy.integrate import quad

def cal_penalty_u(S, u_max, u_min, u):

    class IntegralPenalty(Function):
        @staticmethod
        def forward(ctx, S, u_max, u_min, u):
            ctx.save_for_backward(S, u_max, u_min, u)
            def dpdt(rate,u,u_min=u_min, u_max=u_max, S=S, ):
                u=u.unsqueeze(0)
                u_mid = (u_min+u_max)/2
                s = u_mid + rate*(u - u_mid)
                U = torch.diag((u_max- u_min)/2)
                atanh = lambda x:0.5*torch.log((1+x)/(1-x))
                grad = atanh(
                    U.inverse().mm(
                        (s - u_mid).t()
                    )
                ).t().mm(U).mm(S)*(u-u_mid)
                grad = grad.sum(1)
                return grad

            #self.real_u = self.u_min + self.U*(u+1)
            #penalty_u = quad(dpdt, 0, 1)
            penalty_u = torch.FloatTensor(
                [quad(dpdt,0,1,args=(u_i))[0] for u_i in u]
            )
            return penalty_u

        @staticmethod
        def backward(ctx, grad_output):
            S, u_max, u_min, u = ctx.saved_tensors
            U = torch.diag((u_max- u_min)/2)
            u_mid = (u_min+u_max)/2
            atanh = lambda x:0.5*torch.log((1+x)/(1-x))
            grad = atanh(
                U.inverse().mm(
                    (u - u_mid).t()
                )
            ).t().mm(U).mm(S)

            return None, None, None, (grad.t()*grad_output).t()

    U = torch.diag((u_max- u_min)/2)
    real_u = u_min + (u+1)*0.5*(u_max - u_min)
    penalty_func = IntegralPenalty.apply
    return penalty_func(S, u_max, u_min, real_u)

def integral_test2():
    time_beg = time.time()
    S = torch.FloatTensor([[10,0],[0,10]])
    u_max = torch.FloatTensor([10,10])
    u_min = torch.FloatTensor([-10,-10])
    u = torch.FloatTensor([[3,9],[1,-8],[0,0],[-5,-5]])
    u = 2*(u-u_min)/(u_max-u_min) - 1
    u.requires_grad = True
    p = cal_penalty_u(S, u_max, u_min, u)
    p.backward(torch.ones(p.shape))
    print(u.grad)
    print(time.time() - time_beg)

#integral_test2()