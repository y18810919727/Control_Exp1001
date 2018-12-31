
# -*- coding:utf8 -*-
import numpy as np
import pprint

# 实验类：用于调度env和controller进行交互，并每隔一定的训练轮次(rounds)，对模型进行一次评估
class BaseExp:
    def __init__(self, env=None, controller=None,
                 max_frame=100000,
                 rounds=100,
                 max_step=1000,
                 eval_rounds=0,
                 eval_length=None,
                 exp_name=None):
        """

        :param env: 仿真环境
        :param controller: 控制模型
        :param max_frame: 最大迭代次数
        :param rounds: 训练回合数
        :param max_step: 每个回合内的最大迭代次数
        :param eval_rounds: 评估轮次
        :param eval_length: 每次评估过程中，迭代次数
        :param exp_name: 实验名称
        """

        if eval_rounds == 0 :
            eval_rounds = 1

        if rounds % eval_rounds != 0:
            raise ValueError("rounds should be divided by eval_rounds")

        self.env = env
        self.controller = controller
        self.rounds = rounds
        # 每个round的迭代次数
        self.max_step = max_step
        self.eval_rounds = eval_rounds
        self.eval_length = eval_length
        # 总迭代次数上限
        self.max_frame = max_frame
        self.render_mode = False
        self.log = {}
        if self.eval_length is None:
            self.eval_length = max_step
        if exp_name is None:
            exp_name = "None"
        self.exp_name = exp_name

    def add_log(self,key, value):
        self.log[key] = value

    def render(self):

        print('************Exp**************')
        #print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print('************Exp**************')
        print()

    # 模型评估
    def evaluate(self, t):
        # 将模型的step归零
        self.controller.step_reset()

        # 关闭训练模式，不产生噪音
        self.controller.train_mode = False

        # 重置环境
        s = self.env.reset()
        y_star_list = []
        y_list = []
        reward_list = []

        # 记录最初状态
        y_star_list.append(self.env.y_star[np.newaxis, :])
        # 记录控制结果
        y_list.append(self.env.y[np.newaxis, :])

        for _ in range(self.eval_length):
            # 生成控制指令
            action = self.controller.act(s)
            # 调用仿真
            next_state, r, done, _ = self.env.step(action)
            s = next_state
            reward_list.append(r)
            # 记录目标值
            y_star_list.append(self.env.y_star[np.newaxis, :])
            # 记录控制结果
            y_list.append(self.env.y[np.newaxis, :])

            self.log = {}
            self.add_log("eval_time", (t, _))
            self.add_log("y_star", self.env.y_star)
            self.add_log("y", self.env.y)
            self.add_log("r", r)
            if self.render_mode:
                self.render()

        # 列表转array，每行代表一个控制时刻的结果
        y_star_array = np.concatenate(y_star_list)
        y_array = np.concatenate(y_list)

        # 恢复训练模式
        self.controller.train_mode = True

        return (y_array, y_star_array), reward_list

    def run(self):
        rewards = []
        eval_list = []
        eval_reward_list = []
        # 训练eval_cycle个round之后，进行一次模型评估
        eval_cycle = int(self.rounds/self.eval_rounds)
        frame = 0
        for round_i in range(self.rounds):
            #print(round_i)

            print("Exp :%s, Current round: %i" % ( self.exp_name, round_i))
            state = self.env.reset()
            self.controller.step_reset()
            reward_sum = 0
            for step in range(self.max_step):

                # 控制器计算策略
                action = self.controller.act(state)
                # 仿真环境进行反馈
                next_state, r, done, _ = self.env.step(action)
                # 训练模型
                self.controller.train(state, action, next_state, r, done)
                state = next_state
                reward_sum += r
                frame += 1
                self.log = {}
                self.add_log("frame", frame)
                self.add_log("step", step)
                self.add_log("round", round_i)
                self.add_log("eval_cycle", eval_cycle)
                if self.render_mode:
                    self.render()
                if done:
                    break
                if frame > self.max_frame:
                    break
            rewards.append(reward_sum)

            # 每经过eval_cycle 进行一次模型评估
            if round_i % eval_cycle == 0:
                eval_res = self.evaluate(
                        int(round_i/eval_cycle)
                    )
                eval_list.append(
                    eval_res[0]
                )

                eval_reward_list.append(
                    eval_res[1]
                )

            if frame > self.max_frame:
                break

        # 全部训练完，再评估一次
        if self.rounds == 0 :
            eval_t = 0
        else:
            eval_t = self.rounds/eval_cycle
        eval_res = self.evaluate(
            eval_t
        )
        eval_list.append(
            eval_res[0]
        )

        eval_reward_list.append(
            eval_res[1]
        )
        return rewards, eval_list, eval_reward_list


