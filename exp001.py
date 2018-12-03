import numpy as np
import random
from simulation.demo_simmulation import DemoSimulation as Env
from rewards.demo_reward import DemoReward as Reward
if __name__ == '__main__':
    env = Env(y_high=np.array([12,12],dtype=float),y_low=np.array([8,8],dtype=float),reward_calculator=Reward())
    env.reset()
    t = 10
    while t:
        t=t-1
        u = np.array([random.uniform(-5,5),random.uniform(-2,2)],dtype=float)
        state, reward, done, _ = env.step(u)
        #print(state,reward,done)
        env.render()

        if done:
            env.reset()


