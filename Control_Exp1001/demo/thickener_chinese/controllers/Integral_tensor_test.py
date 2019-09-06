#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torchdiffeq import odeint as odeint_torch
from torch import nn



class Cooling_law_module(nn.Module):
    def __init__(self, a, H):
        super(Cooling_law_module, self).__init__()
        self.a = torch.tensor(a)
        self.a.requires_grad = True
        #self.a = a
        self.H = H

    def forward(self, t, w):
        return -self.a*(w-self.H)


def ode_scipy():
    # -*- coding:utf-8 -*-


    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    import numpy as np

    from IPython import display

    # 冷却定律的微分方程
    def cooling_law_equ(w, t, a, H):
        return -1 * a * (w - H)

    # 冷却定律求解得到的温度temp关于时间t的函数
    def cooling_law_func(t, a, H, T0):
        return H + (T0 - H) * np.e ** (-a * t)

    def ode_array(initial_temp):
        law_func = Cooling_law_module(0.5, 30)
        t = np.arange(0, 10, 0.01)
        temp2 = odeint_torch(law_func, initial_temp, torch.FloatTensor(t))
        return temp2[-1]

    law_func = Cooling_law_module(0.5, 30)
    t = np.arange(0, 10, 0.01)
    initial_temp = (90) #初始温度
    temp = odeint(cooling_law_equ, initial_temp, t, args=(0.5, 30)) #冷却系数和环境温度
    temp1 = cooling_law_func(t, 0.5, 30, initial_temp) #推导的函数与scipy计算的结果对比
    temp2 = odeint_torch(law_func, torch.Tensor([initial_temp]), torch.FloatTensor(t))
    temp2  = temp2.squeeze(1)
    plt.subplot(3, 1, 1)
    plt.plot(t, temp)
    plt.ylabel("temperature")

    plt.subplot(3, 1, 2)
    plt.plot(t, temp1)
    plt.xlabel("time")
    plt.ylabel("temperature")

    plt.subplot(3, 1, 3)
    plt.plot(t, np.array(temp2.detach().numpy()))
    plt.xlabel("time")
    plt.ylabel("temperature")

    display.Latex("牛顿冷却定律 $T'(t)=-a(T(t)- H)$)（上）和 $T(t)=H+(T_0-H)e^{-at}$（下）")
    plt.show()
    input_data = torch.FloatTensor([initial_temp])
    input_data.requires_grad = True
    torch.autograd.gradcheck(ode_array, input_data)


def cal_penalty_u(S, u_max, u_min, u, time_sep=0.001):

    class Dpdt(nn.Module):
        def __init__(self, S, u_max, u_min, u):
            super(Dpdt, self).__init__()
            self.S = torch.FloatTensor(S)
            self.u_max = torch.FloatTensor(u_max)
            self.u_min = torch.FloatTensor(u_min)
            self.U = torch.diag((self.u_max- self.u_min)/2)
            self.mid_u = (self.u_max+self.u_min)/2
            #self.real_u = self.u_min + self.U*(u+1)
            self.real_u = u

        def atanh(self, x):
            return 0.5*torch.log((1+x)/(1-x))
            # return x+1
        def forward(self, t, p0):
            s = self.mid_u + t*(self.real_u-self.mid_u)
            grad = self.atanh(
                self.U.inverse().mm(
                    (s - self.mid_u).t()
                )
            ).t().mm(self.U).mm(self.S)*(self.real_u-self.mid_u)
            grad = grad.sum(1)
            return grad
    dpdt = Dpdt(S, u_max, u_min, u)
    t = np.arange(0, 1, time_sep)
    p0 = torch.zeros(u.shape[0])
    penalty = odeint_torch(dpdt, p0, torch.FloatTensor(t))
    return penalty[-1]

def cal_penalty_u2(S, u_max, u_min, u):
    from torch.autograd import Function
    from scipy.integrate import quad

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

    penalty_func = IntegralPenalty.apply
    return penalty_func(S, u_max, u_min, u)

import time
def integral_test(time_sep=0.001, right_tensor=None):
    time_beg = time.time()

    S = torch.FloatTensor([[10,0],[0,10]])
    u_max = torch.FloatTensor([10,10])
    u_min = torch.FloatTensor([-10,-10])
    u = torch.FloatTensor([[3,9],[1,-8],[0,0],[-5,-5]])
    u.requires_grad = True
    p = cal_penalty_u(S, u_max, u_min, u, time_sep=time_sep)
    p.backward(torch.ones(p.shape))
    #print(u.grad)
    print(torch.dist(right_tensor, u.grad))
    # def check_grad_func(u):
    #     S = torch.FloatTensor([[10,0],[0,10]])
    #     u_max = torch.FloatTensor([10,10])
    #     u_min = torch.FloatTensor([-10,-10])
    #     p = cal_penalty_u(S, u_max, u_min, u)
    #     return p
    #
    # torch.autograd.gradcheck(check_grad_func, u)
    print(time.time()-time_beg)

def integral_test2():
    time_beg = time.time()
    S = torch.FloatTensor([[10,0],[0,10]])
    u_max = torch.FloatTensor([10,10])
    u_min = torch.FloatTensor([-10,-10])
    u = torch.FloatTensor([[3,9],[1,-8],[0,0],[-5,-5]])
    u.requires_grad = True
    p = cal_penalty_u2(S, u_max, u_min, u)
    p.backward(torch.ones(p.shape))
    print(u.grad)
    print(time.time() - time_beg)
    return u.grad


right_grad = integral_test2()
integral_test(0.01,right_tensor=right_grad)
integral_test(0.001,right_tensor=right_grad)
integral_test(0.0001,right_tensor=right_grad)
integral_test(0.00001,right_tensor=right_grad)
