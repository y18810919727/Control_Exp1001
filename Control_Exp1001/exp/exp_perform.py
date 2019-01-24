
# -*- coding:utf8 -*-
import threading

import numpy as np


# 每个exp由一个单独的线程运行
def exp_run_thread(exp, pos, res_list):

    print("%s begin" % exp.exp_name)
    res_list[pos]=exp.run()
    np.random.seed()
    print("%s end" % exp.exp_name)


def exp_multi_thread_run(exp_list):
    res_list = [None for i in range(len(exp_list))]
    thread_t = []
    for i in range(len(exp_list)):
        # 每个实验块创建一个线程
        t = threading.Thread(name="task %s" % exp_list[i].exp_name, target=exp_run_thread, args=[exp_list[i], i, res_list])
        thread_t.append(t)
        t.start()

    # 阻塞主线程
    for t in thread_t:
        t.join()

    print("All exp stops")
    return res_list


def exp_single_thread_run(exp_list):
    res_list = [exp_list[i].run() for i in range(len(exp_list))]
    print("All exp stops")
    return res_list
