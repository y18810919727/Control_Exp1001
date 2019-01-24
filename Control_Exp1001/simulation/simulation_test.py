#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np

from Control_Exp1001.common.evaluation.one_round_evaluation import OneRoundEvaluation as EvaluationBase
from Control_Exp1001.control.base_control import ControlBase
from Control_Exp1001.exp.exp_perform import exp_single_thread_run
from Control_Exp1001.exp.one_round_exp import OneRoundExp as BaseExp


class Test_Control(ControlBase):
    def __init__(self, u_bounds, act_u=None):
        super(Test_Control, self).__init__(u_bounds)
        self.act_u = act_u
        self.u_bounds = u_bounds

    def _act(self, state):
        return self.act_u(state)

    def train(self, s, u, ns, r, done):
        pass

def test_controller_construct(act_list, Env,init_para=None, act_name=None, test_step=200):

    exp_list = []
    if act_name is None:
        act_name = [str(i+1) for i in range(len(act_list))]
    for id, act in enumerate(act_list):
        env = Env(**init_para)
        controller = Test_Control(np.copy(env.u_bounds), act_u=act)

        exp = BaseExp(env=env, controller=controller,
                      exp_name=act_name[id], max_step=test_step)
        exp_list.append(exp)


    return exp_list

def simulation_test(Env=None, mode="uniform",init_para=None,
                    seprate_num=5, const_u=None,
                    test_step=200, y_name=None,
                    eval_plt_param=None,
                    ):
    if Env is None:
        raise ValueError("No env to simulation")
    if init_para is None:
        init_para = {}
    tmp_env = Env(**init_para)
    u_bounds = tmp_env.external_u_bounds

    low = u_bounds[:, 0]
    high = u_bounds[:, 1]
    act_name = []
    if mode is "uniform":
        act_u_list = []
        for id in np.arange(0, seprate_num,1):
            action = np.copy((high-low)/seprate_num*id) + low
            act_u = np.copy(action)
            act_u_list.append(lambda x, act=act_u: act)
            act_name.append(str(action))
        act_u_list.append(lambda x, act=high: act)
        act_name.append(str(high))

        exp_list = test_controller_construct(act_u_list, Env,init_para=init_para, act_name=act_name, test_step=test_step)


    elif mode is "random":
        #act_u = np.copy(np.random.uniform(low, high))
        act_u = lambda x, al=low, ah=high: np.random.uniform(al, ah)
        act_name = ["random action"]
        exp_list = test_controller_construct([act_u], Env, init_para=init_para, act_name=act_name, test_step=test_step)

    elif mode is "const":
        if const_u is None:
            raise ValueError("const_u could not be None")
        act_u_list = []
        act_name = []
        for id, u in enumerate(const_u):
            act_u = np.array(u)
            act_u_list.append(lambda x, act=act_u: act)
            act_name.append(str(act_u))
        exp_list = test_controller_construct(act_u_list, Env,init_para=init_para, act_name=act_name, test_step=test_step)

    else:
        raise ValueError("mode should be assigned in {uniform, const, random}")

    res_list = exp_single_thread_run(exp_list)
    evaluation = EvaluationBase(res_list=res_list)
    evaluation.plot_all()
