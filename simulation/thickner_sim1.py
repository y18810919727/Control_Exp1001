import numpy as np
import matplotlib.pyplot as plt
from simulation.base_env import BaseEnv
import random

class ThickenerSimulation(BaseEnv):

    def __init__(self, dt=1, reward_calculator=None,
                 size_yudc=None,
                 y_low=None, u_low=None,
                 d_low=None, c_low=None,
                 y_high=None, u_high=None,
                 d_high=None, c_high=None,
                 normalize=True,
                 step_length = 0.1,
                 step_seg = 1000,
                 ):
        if size_yudc is None:
            size_yudc = [2, 2, 0, 2]

        super(ThickenerSimulation, self).__init__(dt, reward_calculator, size_yudc,
                                             y_low, u_low,
                                             d_low, c_low,
                                             y_high, u_high,
                                             d_high, c_high, normalize)
        self.step_length = step_length
        self.step_seg = step_seg
        self.y_begin = np.array([10, 10], dtype=float)
        self.u_begin = np.array([0, 0], dtype=float)
        self.rho_s = 4150
        self.rho_e = 1803
        self.mu_e = 0.05
        self.d0 = 0.0008
        self.p = 0.5
        self.A = 1937.5
        self.ks = 0.0001
        self.ki = 0.0005
        self.Ki = 50
        self.Ku = 2
        self.Kf = 0.75
        self.theta = 85
        self.g = 9.8
        self.y_begin = np.array([1.38, 680],dtype=float)
        self.u_begin = np.array([35,58],dtype=float)
        self.c_begin = np.array([40, 73], dtype=float)

    def observation(self):
        return np.concatenate((self.y_star, self.y, self.u, self.c))

    def reset_y(self):

        return self.y_begin + np.array([random.uniform(-0.3, 0.3),random.uniform(-100, 100)], dtype=float)

    def reset_y_star(self):
        return np.array([1.44, 700], dtype=float)

    def reset_u(self):
        return self.u_begin + np.array([random.uniform(-7, 7), random.uniform(-7, 7)], dtype=float)

    def reset_c(self):
        #return np.random.multivariate_normal(np.zeros(2),np.diag(np.ones(2)))
        return self.c_begin + np.array([random.uniform(-4, 4), random.uniform(-30, 30)], dtype=float)

    def f(self, y, u, c, d):
        ht, cu = tuple(y.tolist())
        fu, ff = tuple(u.tolist())
        fi, ci = tuple(c.tolist())

        qi = self.Ki * fi
        qu = self.Ku * fu
        qf = self.Kf * ff
        dt = self.ks*qf + self.d0
        ut = dt*dt*(self.rho_s-self.rho_e)*self.g/(18*self.mu_e)

        for _ in range(self.step_seg):

            (grad_cu, grad_ht) = self.cal_grad(self.ki, qi, ci, ut, self.A, cu, qu, self.p, ht, self.theta, self.rho_s)
            cu = cu + grad_cu * float(self.step_length)/self.step_seg
            ht = ht + grad_ht * float(self.step_length)/self.step_seg

        y = np.array([ht, cu],dtype=float)
        #c = c + u[0]

        return y, u, c, d

    @staticmethod
    def cal_grad(ki,qi,ci,ut,A3,cu,qu,p,h3,theta,rho):
        x1 = ki*qi*ci*ut*A3
        x2 = x1 + (ki*qi*ci-cu)*qu
        x3 = x2*(ki*qi*ci+cu)
        x4 = p*h3*A3*(ki*qi*ci+cu)-ci*qi*theta*rho
        y1=ci*qi*theta*rho*(ki*qi*ci*ut*A3+(ki*qi*ci-cu)*qu)
        y2 = -A3*p*(ki*qi*ci+cu)*(p*h3*A3*(ki*qi*ci+cu)-ci*qi*theta*rho)

        return x3/x4,y1/y2

if __name__ == '__main__':

    rounds = 1
    times = 500000
    step_seg = 1
    step_length = 1

    sim = ThickenerSimulation(
        step_length=step_length,
        step_seg=step_seg,
        normalize=False
    )
    y0_list = []
    y1_list = []
    for _ in range(rounds):
        sim.reset()
        y0_list = []
        y1_list = []
        for i in range(times):
            sim.step()
            y0_list.append(sim.y[0])
            y1_list.append(sim.y[1])
        x_list = np.arange(0, step_length*times, step_length)
        plt.subplot(211)
        plt.plot(x_list, y0_list)
        plt.ylabel("Height")

        plt.xlabel("time")
        plt.subplot(212)
        plt.plot(x_list, y1_list)
        plt.ylabel("Cu")
        plt.xlabel("time")
        plt.show()



