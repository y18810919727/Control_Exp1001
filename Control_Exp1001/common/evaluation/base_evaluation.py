
# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd

class EvaluationBase():
    def __init__(self, res_list,training_rounds=None,
                 exp_name=None, y_name=None, y_num=None,
                 penalty_plt_param=None,
                 eval_plt_param=None,
                 ):
        """

        :param res_list: (奖赏变化, 评估结果, 评估过程奖赏)
        :param exp_name: 实验名称列表
        :param y_name: y的名称
        :param training_rounds: 训练轮次
        :param y_num: y的维度大小
        :param penalty_plt_param: 画penalty图的plt参数
        :param eval_plt_param: 画评估图的plt参数
        """

        # 解压实验结果，分别是奖赏变化,训练过程中评估结果，训练过程评估时的奖赏值
        self.penaltys_list, self.eval_list, self.eval_penalty_list = zip(*res_list)
        if y_num is None:
            raise ValueError("y_num shouldn't be None")
        self.y_num = y_num
        if exp_name is None:
            exp_name = ["exp"+str(i+1) for i in range(len(res_list))]
        self.exp_name = exp_name
        if training_rounds is None:
            training_rounds = len(self.penaltys_list[0])
        self.training_rounds = training_rounds

        if y_name is None:
            y_name = ["y"+str(i+1) for i in range(self.y_num)]
        self.y_name = y_name

        if penalty_plt_param is None:
            penalty_plt_param = {}
        if eval_plt_param is None:
            eval_plt_param = {}

        self.eval_plt_param = eval_plt_param
        self.penalty_plt_param = penalty_plt_param

    def draw_penaltys(self):

        plt.figure(**self.penalty_plt_param)
        plt.title("Penaltys in each round")
        for id, penaltys in enumerate(self.penaltys_list):
            #plt.figure()
            plt.plot(penaltys)

        plt.legend(self.exp_name)
        plt.xlabel("Rounds")
        plt.ylabel("Penaltys")
        plt.show()

    def draw_eval(self, draw_title=True):

        # 直接将评估结果转换为nparray
        eval_array = np.array(self.eval_list)
        eval_penalty_list = np.array(self.eval_penalty_list)
        exp_num, rounds, _, time, y_num = eval_array.shape
        # eval_array.shape :[exp id, round id, y or y* or r, time, dim of y]
        for round_id in range(rounds):
            # 输出每个实验块在该此评估的奖赏和
            for exp_id in range(exp_num):
                print("%s peformed %f in eval round %i" % (self.exp_name[exp_id], np.sum(eval_penalty_list[exp_id, round_id,:]), round_id))
            figure = plt.figure(**self.eval_plt_param)
            if draw_title:

                if rounds > 1:
                    # -1是为了减去最后的那次评估
                    figure.suptitle("After %i training tounds" % (round_id*self.training_rounds/(rounds-1)))
                else:
                    figure.suptitle("After %i training tounds" % 0)

            #figure.suptitle("Training Rounds %i" % (round_id*self.training_rounds/(rounds-1)),fontsize=16,x=0.53,y=1.05)
            #plt.title("Training Rounds %i" % round_id)
            # A set of curves for one y
            # 画出控制效果图
            for yid in range(y_num):
                plt.subplot(y_num,1,yid+1)
                plt.plot(eval_array[0, round_id, 1, :, yid])
                for exp_id in range(exp_num):
                    plt.plot(eval_array[exp_id, round_id, 0, :, yid])

                legend_name = ["set point"]+self.exp_name
                plt.legend(legend_name)
                #plt.text("Acc penalty")
                plt.ylabel(self.y_name[yid])
                plt.xlabel("time")
            plt.show()

    def error_eval(self):
        eval_array = np.array(self.eval_list)
        eval_penalty_list = np.array(self.eval_penalty_list)
        exp_num, rounds, _, time, y_num = eval_array.shape
        # eval_array.shape :[exp id, round id, y or y* or r, time, dim of y]
        err_res = []

        for round_id in range(rounds):
            # 输出每个实验块在该此评估的奖赏和
            for yid in range(y_num):
                error_array = np.zeros((2,exp_num))
                print("------Round:%i, Y:%s----------" % (round_id, self.y_name[yid]))
                for exp_id in range(exp_num):
                    y = (eval_array[exp_id, round_id, 0, :, yid]).reshape(-1,1)
                    y_star = (eval_array[exp_id, round_id, 1, :, yid]).reshape(-1,1)

                    # 计算控制误差
                    mse = mean_squared_error(y,y_star)
                    mae = mean_absolute_error(y,y_star)

                    error_array[0, exp_id] = mse
                    error_array[1, exp_id] = mae

                # 转换为pandas.DataFrame，输出好看
                res_df = pd.DataFrame(error_array,index=["MSE", "MAE"], columns=self.exp_name)
                err_res.append(res_df)

                print(res_df)
                print("------E      N      D----------" )
                print()

        return err_res



