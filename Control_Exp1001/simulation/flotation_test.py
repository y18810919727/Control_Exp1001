#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import Control_Exp1001 as CE
from Control_Exp1001.simulation.flotation import Flotation
from Control_Exp1001.simulation.simulation_test import simulation_test


para = {
    "normalize": True,
}
simulation_test(Flotation, init_para=para, mode="const",
                const_u=[[0, 0], [-0.5, 0.5], [0.5, 0.5]],
                test_step=100, eval_plt_param={"figsize": (15, 10)})

simulation_test(Flotation,  mode="const",
                const_u=[[1, 17], [1.5, 17], [2, 3], [2, 20], [2.5, 17]] , seprate_num=3,
                test_step=100, eval_plt_param={"figsize": (15, 10)})
#simulation_test(Flotation, mode="random", test_step=100, eval_plt_param={"figsize": (15, 10)})

simulation_test(Flotation, init_para=para, mode="uniform", seprate_num=2, test_step=100, eval_plt_param={"figsize": (15, 10)})
simulation_test(Flotation, seprate_num=2, test_step=100, eval_plt_param={"figsize": (15, 10)})
simulation_test(Flotation, init_para=para, mode="random", test_step=100, eval_plt_param={"figsize": (15, 10)})



