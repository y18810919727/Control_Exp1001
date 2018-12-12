import numpy as np
import matplotlib.pyplot as plt

class EvaluationBase():
    def __init__(self, res_list,training_rounds=None,
                 exp_name=None, y_name=None, y_num=None,
                 reward_plt_param=None,
                 eval_plt_param=None,
                 ):
        """

        :param res_list: (奖赏变化, 评估结果)
        :param exp_name: 实验名称列表
        :param y_name: y的名称
        :param plt_param: 绘图参数
        """
        self.rewards_list, self.eval_list = zip(*res_list)
        if y_num is None:
            raise ValueError("y_num shouldn't be None")
        self.y_num = y_num
        if exp_name is None:
            exp_name = ["exp"+str(i+1) for i in range(len(res_list))]
        self.exp_name = exp_name
        if training_rounds is None:
            training_rounds = len(self.rewards_list[0])
        self.training_rounds = training_rounds

        if y_name is None:
            y_name = ["y"+str(i+1) for i in range(self.y_num)]
        self.y_name = y_name

        if reward_plt_param is None:
            reward_plt_param = {}
        if eval_plt_param is None:
            eval_plt_param = {}

        self.eval_plt_param = eval_plt_param
        self.reward_plt_param = reward_plt_param

    def draw_rewards(self):

        plt.figure(**self.reward_plt_param)
        plt.title("Rewards in each round")
        for id, rewards in enumerate(self.rewards_list):
            #plt.figure()
            plt.plot(rewards)

        plt.legend(self.exp_name)
        plt.xlabel("Rounds")
        plt.ylabel("Rewards")
        plt.show()

    def draw_eval(self):

        eval_array = np.array(self.eval_list)
        exp_num, rounds, _, time, y_num = eval_array.shape
        # eval_array.shape :[exp, round, y or y*, time, dim of y]
        for round_id in range(rounds):
            figure = plt.figure(**self.eval_plt_param)
            figure.suptitle("Training Rounds %i" % (round_id*self.training_rounds/(rounds-1)))
            #figure.suptitle("Training Rounds %i" % (round_id*self.training_rounds/(rounds-1)),fontsize=16,x=0.53,y=1.05)
            #plt.title("Training Rounds %i" % round_id)
            # A set of curves for one y
            for yid in range(y_num):
                plt.subplot(y_num,1,yid+1)
                plt.plot(eval_array[0, round_id, 1, :, yid])
                for exp_id in range(exp_num):
                    plt.plot(eval_array[exp_id, round_id, 0, :, yid])
                legend_name = ["set point"]+self.exp_name
                plt.legend(legend_name)
                plt.ylabel(self.y_name[yid])
                plt.xlabel("time")
            plt.show()

