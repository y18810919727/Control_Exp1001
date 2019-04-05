# -*- coding:utf8 -*-

class OneRoundEvaluation():
    def __init__(self, res_list
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
        # self.penaltys_list, self.eval_list, self.eval_penalty_list = zip(*res_list)
        self.y_list, self.u_list, self.c_list, self.d_list, self.penalty_list, self.u0_grad, self.u1_grad, \
        self.y0_grad, self.y1_grad = zip(*res_list)

        # 将每个实验块跑出来的结果合并。
        self.y_data = self.y_list[0].merge_list(self.y_list)
        self.u_data = self.u_list[0].merge_list(self.u_list)
        self.c_data = self.c_list[0].merge_list(self.c_list)
        self.d_data = self.d_list[0].merge_list(self.d_list)
        self.penalty_data = self.penalty_list[0].merge_list(self.penalty_list)
        self.grad_data = list()
        for x in [self.u0_grad, self.u1_grad, self.y0_grad, self.y1_grad]:
            self.grad_data.append(x[0].merge_list(x))

    def plot_all(self):
        self.y_data.plt()
        mse_dict = self.y_data.cal_mse()
        self.u_data.plt()
        self.c_data.plt()
        self.d_data.plt()
        # for x in self.grad_data:
        #     x.plt()

        self.penalty_data.plt()
        return mse_dict
