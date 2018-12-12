import numpy as np
import pprint

class BaseExp:
    def __init__(self, env=None, controller=None,
                 max_frame=100000,
                 rounds=100,
                 max_step=1000,
                 eval_rounds=10,
                 eval_length=None):

        if eval_rounds == 0 :
            eval_rounds = 1

        if rounds % eval_rounds != 0:
            raise ValueError("rounds should be divided by eval_rounds")

        self.env = env
        self.controller = controller
        self.rounds = rounds
        self.max_step = max_step
        self.eval_rounds = eval_rounds
        self.eval_length = eval_length
        self.max_frame = max_frame
        self.render_mode = False
        self.log = {}
        if self.eval_length is None:
            self.eval_length = max_step

    def add_log(self,key, value):
        self.log[key] = value

    def render(self):

        print('************Exp**************')
        #print("Step : %i" % self.step)
        pprint.pprint(self.log)
        print('************Exp**************')
        print()

    def evaluate(self, t):
        self.controller.step_reset()
        self.controller.train_mode = False
        s = self.env.reset()
        y_star_list = []
        y_list = []
        for _ in range(self.eval_length):
            action = self.controller.act(s)
            next_state, r, done, _ = self.env.step(action)
            s = next_state
            y_star_list.append(self.env.y_star[np.newaxis, :])
            y_list.append(self.env.y[np.newaxis, :])

            self.log = {}
            self.add_log("eval_time", (t, _))
            self.add_log("y_star", self.env.y_star)
            self.add_log("y", self.env.y)
            self.add_log("r", r)
            if self.render_mode:
                self.render()

        y_star_array = np.concatenate(y_star_list)
        y_array = np.concatenate(y_list)

        # 恢复训练模式
        self.controller.train_mode = True


        return y_array, y_star_array

    def run(self):
        rewards = []
        eval = []
        eval_cycle = int(self.rounds/self.eval_rounds)
        frame = 0
        for round_i in range(self.rounds):
            print(round_i)
            state = self.env.reset()
            self.controller.step_reset()
            reward_sum = 0
            for step in range(self.max_step):
                action = self.controller.act(state)
                next_state, r, done, _ = self.env.step(action)
                self.controller.train(state, action, next_state, r, done)
                state = next_state
                reward_sum += r
                if done:
                    break
                frame += 1
                if frame > self.max_frame:
                    break
                self.log = {}
                self.add_log("frame", frame)
                self.add_log("step", step)
                self.add_log("round", round_i)
                self.add_log("eval_cycle", eval_cycle)
                if self.render_mode:
                    self.render()
            rewards.append(reward_sum)

            if round_i % eval_cycle == 0:
                eval.append(
                    self.evaluate(
                        int(round_i/eval_cycle)
                    )
                )

            if frame > self.max_frame:
                break

        eval.append(
            self.evaluate(
                int(self.rounds/eval_cycle)
            )
        )
        return rewards, eval












