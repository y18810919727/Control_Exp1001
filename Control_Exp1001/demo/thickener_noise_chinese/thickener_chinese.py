#!/usr/bin/python

import random
import sys

# -*- coding:utf8 -*-
import numpy as np
from scipy.integrate import odeint

from Control_Exp1001.simulation.base_env import BaseEnv
from Control_Exp1001.simulation.simulation_test import simulation_test


class Thickener(BaseEnv):
    def __init__(self, dt=1, penalty_calculator=None, noise_in=False, noise_p=0.01,
                 size_yudc=None, u_low=None,
                 u_high=None,
                 c_low=None,
                 c_high=None,
                 y_low=None,
                 y_high=None,
                 normalize=False,
                 time_length=120,
                 one_step_length=0.0005,
                 y_name=None,
                 u_name=None,
                 c_name=None,
                 c_start=None,
                 y_start=None,
                 y_star=None,
                 mean_c=None,
                 cov_c=None,
                 random_seed=None,
                 noise_type=None
                 ):
        """

        :param dt:
        :param penalty_calculator:
        :param noise_in: 是否使进料浓度和流量波动
        :param size_yudc:
        :param u_low:
        :param u_high:
        :param normalize:
        :param time_length: 默认一次仿真120秒
        :param one_step_length:
        :param y_name: 默认泥层高度和 底流浓度
        :param c_name: 默认进料泵速和进料单位体积含固量密度
        :param y_star:
        :param noise_type: 1:第400次突变2:随机波动噪音，3：1600时突变
        """
        if size_yudc is None:
            size_yudc = [2, 2, 0, 2]
        if y_name is None:
            y_name = ["泥层高度", "底流浓度"]
        if c_name is None:
            c_name = ["进料泵速", "进料浓度"]
        if u_name is None:
            u_name = ["底流泵速", "絮凝剂泵速"]

        if c_start is None:
            c_start = np.array([40, 73], dtype=float)
        self.c_start = np.array(c_start)

        if y_start is None:
            y_start = np.array([1.5, 660], dtype=float)
        self.y_start = np.array(y_start)

        if u_low is None:
            u_low = [40, 30]
        if u_high is None:
            u_high = [120, 50]

        if c_low is None:
            c_low = [34, 63]
        if c_high is None:
            c_high = [46, 83]

        if y_low is None:
            y_low = [0.75, 280]
        if y_high is None:
            y_high = [2.5, 1200]
        self.noise_in = noise_in

        if mean_c is None:
            mean_c = np.copy(c_start)
        if cov_c is None:
            cov_c = np.array([[10, 0], [0, 18]])
        self.mean_c = np.array(mean_c)
        self.cov_c = np.array(cov_c)
        self.noise_p = noise_p
        self.random_seed = random_seed
        self.noise_type = noise_type
        super(Thickener, self).__init__(
            dt=dt,
            penalty_calculator=penalty_calculator,
            size_yudc=size_yudc,
            u_low=u_low,
            u_high=u_high,
            c_low=c_low,
            c_high=c_high,
            y_low=y_low,
            y_high=y_high,

            normalize=normalize,
            time_length=time_length,
            one_step_length=one_step_length,
            y_name=y_name,
            u_name=u_name,
            c_name=c_name,
        )

        if y_star is None:
            #y_star = np.array([1.48, 680], dtype=float)
            y_star = np.array([1.48, 680], dtype=float)
        self.y_star = np.array(y_star)


        self.param = {}

        # Todo 密度好像大了点
        self.param['rho_s'] = 4150  # 原文砂密度

        self.param['rho_e'] = 1803  # 介质表观密度
        self.param['rho_l'] = 1000  # 介质表观密度
        self.param['mu_e'] = 2  # 介质表观粘度 0.01-1 ????
        self.param['d0'] = 0.00008  # 进口处颗粒直径
        self.param['p'] = 0.5  # 平均浓度系数
        # self.param['A'] = 1937.5  # 浓密机底面积
        self.param['A'] = 300.5  # 原浓密机底面积过大，导致整体下行速度太慢了，粒子速度基本等于由修正后的斯托克斯计算的速度

        # Todo 这个参数好像有点大，导致粒子太大，下沉太快
        # self.param['ks'] = 0.36  # 絮凝剂作用系数，相比原文统一单位乘了3600,
        self.param['ks'] = 0.157  # 絮凝剂作用系数，ut还是大，再让它小点

        self.param['ki'] = 0.0005 * 3600  # 压缩层浓度系数，会影响泥层界面高度处的浓度，
        self.param['Ki'] = 50.0 / 3600  # 进料流量(m^3/ s) 与进料泵速(单位：Hz，约为40) 的比值
        self.param['Ku'] = 2.0 / 3600  # 出料流量(m^3/ s) 与出料泵速(单位：Hz，约为80) 的比值
        self.param['Kf'] = 0.75 / 3600  # 絮凝剂流量(m^3/ s) 与絮凝剂泵泵速(单位：Hz，约为40) 的比值
        # Todo
        self.param['theta'] = 85  # 底流压缩到满足出料要求的液固比所需要的时间,原文85秒太短了
        self.param['theta'] = 3000  # 增大！
        self.param['theta'] = 2300  # 24 Jan更新，解决增加进料噪音后b-a*d大于0的问题
        self.param['g'] = 9.8

    def reset_u(self):
        return np.array([80, 38], dtype=float)

    def reset_y(self):
        return np.copy(self.y_start)

    def reset_c(self):
        # new_c = np.array([40, 73], dtype=float)
        return np.copy(self.c_start)

    def reset_y_star(self):
        return np.copy(self.y_star)

    def observation(self):
        return np.hstack([self.y_star, self.y, self.u, self.c])

    def f(self, y, u, c, d):
        # 当前(高度, 底流浓度)
        ht, cu = tuple(y.tolist())
        # 控制量(底流泵频, 絮凝剂频率)
        fu, ff = tuple(u.tolist())
        # 当前(进料泵速, 进料浓度)
        fi, ci = tuple(c.tolist())

        # region ODE tool
        t_array = np.linspace(0, self.time_length, int(self.time_length / self.one_step_length))
        y_begin = y.tolist()
        # 调用scipy中计算常微分方程的工具
        y_new = odeint(self.cal_grad_inter, y_begin, t_array, args=(fu, ff, fi, ci, self.param,), )

        y = np.copy(y_new[-1, :])
        # endregion

        # region my ODE 后来发现scipy解ode的工具跑得快得多，不用自己写的了
        # for _ in np.arange(0, self.time_length, self.one_step_length):
        #
        #     (grad_cu, grad_ht) = self.cal_grad(
        #         y,
        #         fu, ff,
        #         fi, ci,
        #         **self.param
        #     )
        #     cu = cu + self.one_step_length * grad_cu
        #     ht = ht + self.one_step_length * grad_ht
        #
        # y = np.array([ht, cu],dtype=float)
        # endregion

        # print(self.time_step, 'u',u,"c",c,"ht", 'old y',[ht2,cu2], 'new y', y)

        # region update c
        c = self.update_c(c)
        # endregion

        return y, u, c, d

    def update_c(self, c):

        if self.noise_type is None:

            if self.noise_in and np.random.uniform(0,1) < self.noise_p:
                # print('*'*20)
                # print('update c')
                # print('*'*20)
                c = np.random.multivariate_normal(mean=self.mean_c, cov=self.cov_c)
                # 限制c的上下限
                c = self.bound_detect(c, self.c_bounds)[2]
        elif self.noise_type == 1:
            if self.time_step == 400:
                c = np.array([35, 65])
        elif self.noise_type == 2:

            if self.time_step%6==0:
                np.random.seed(self.random_seed)
                det_c_mean = (self.c_bounds[:, 0] + self.c_bounds[:, 1])/2 - c
                c = c + np.random.multivariate_normal(mean=0.001*det_c_mean, cov=[[0.6,0],[0,0.6]])
                self.random_seed = np.random.randint(0,int(1e9))
                c = self.bound_detect(c, self.c_bounds)[2]

        elif self.noise_type == 3:
            if self.time_step == 800:
                c = np.array([35, 65])
        return c
    # 用那个常微分工具，不能直接在args中写**dict,建立一个中转的静态方法
    @staticmethod
    def cal_grad_inter(y, t, fu, ff, fi, ci, para):
        para['y'] = y
        para['fu'] = fu
        para['ff'] = ff
        para['fi'] = fi
        para['ci'] = ci
        return Thickener.cal_grad(**para)

    @staticmethod
    def cal_grad(
            y,
            fu, ff,
            fi, ci,
            rho_s,
            rho_e,
            rho_l,
            mu_e,
            d0,
            p,
            A,
            ks,
            ki,
            Ki,
            Ku,
            Kf,
            theta,
            g,

    ):
        ht, cu = y
        qi = Ki * fi  # 进料流量(m^3 / s)
        qu = Ku * fu  # 出料流量(m^3 / s)
        qf = Kf * ff  # 絮凝剂流量(m^3 / s)
        dt = ks * qf + d0  # 被絮凝了的粒子大小
        ut = dt * dt * (rho_s - rho_e) * g / (18 * mu_e)  # 修正后的粒子自由沉降速度
        ur = qu / A  # 由于底流导致的总体下行速度，大概是ut的10分之一左右
        cl = ki * qi * ci  # 泥层界面高度处的密度(kg/m^3)
        ca = p * (cl + cu)  # 泥层内平均密度
        wt = ci * qi  # 单位时间进入浓密机的固体质量
        wt_out = cu * qu  # 单位时间进入浓密机的固体质量
        r3 = rho_l * (1 / ca - 1 / rho_s)

        # 定义中间计算变量， 具体含义参看文档

        a = ca
        b = ht
        c = cl * (ut + ur) - cu * ur
        d = wt * theta / A / (ca * ca)

        # 这个assert保证底流泵泵速增大时，底流浓度降低，泥层高度增加
        assert b > a * d

        y = c / (b - a * d)
        x = -d * y

        grad_ca = y
        grad_ht = x
        grad_cu = grad_ca / p

        return [grad_ht, grad_cu]

    def _reset(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)







def debug_const():
    simulation_test(Thickener, mode="const",
                    const_u=[[60, 30], [60, 50], [80, 30], [85, 38], [80, 50], [100, 30], [100, 50]],
                    test_step=120, eval_plt_param={"figsize": (15, 10)})


def debug_cy_mul():
    init_para = {
        "noise_in": True
    }
    simulation_test(Thickener, mode='const', init_para=init_para, test_step=500,
                    const_u=[[60, 30], [60, 50], [80, 30], [85, 38], [80, 50], [100, 30], [100, 50]], )


def debug_cy():
    init_para = {
        "noise_in": True,
        "noise_p": 0.01
    }
    simulation_test(Thickener, mode='const', init_para=init_para, test_step=1000,
                    const_u=[[85, 38]])


def debug_s():
    init_para = {
        "c_start": [46, 77.61991659],
        "y_start": [1.142443791260041, 759.9451550012649]

    }

    simulation_test(Thickener, mode='const', init_para=init_para, test_step=2000, const_u=[[80, 40]])


def debug_random():
    simulation_test(Thickener, init_para={"noise_in": True}, mode="random", test_step=3000,
                    eval_plt_param={"figsize": (15, 10)})


if __name__ == '__main__':

    if len(sys.argv) > 1:
        # 恒定控制，波动输入
        if sys.argv[1] == '-cy':
            debug_cy()
        # 恒定控制，波动输入
        elif sys.argv[1] == '-const':
            debug_const()
        # 指定c y起始值随机策略
        elif sys.argv[1] == '-s':
            debug_s()
        # 多个恒定策略，噪音输入
        elif sys.argv[1] == '-cy_mul':
            debug_cy_mul()
        # 随机策略，噪音输入
        elif sys.argv[1] == '-random':
            debug_random()
