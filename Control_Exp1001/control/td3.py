
# -*- coding:utf8 -*-
from Control_Exp1001.common.network.network_copy import soft_update
from Control_Exp1001.common.network.network_copy import copy_net

from Control_Exp1001.control.base_ac import *


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3, device=None):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_actions+num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.device = device

    def forward(self, state, action):
        x = torch.cat([state,action],1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, device=None):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.device = device

    def forward(self, state):
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # 将输出归到(-1,1)
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


class Td3(ACBase):
    def __init__(self,
                 gpu_id=0,
                 num_inputs=None,
                 num_actions=None,
                 act_hidden_size=None,
                 val_hidden_size=None,
                 replay_buffer=None,
                 u_bounds=None,
                 exploration=None,
                 batch_size=64,
                 policy_lr=1e-3,
                 value_lr=1e-3,
                 noise_std=0.2,
                 noise_clip = 0.5,
                 gamma=0.99,
                 policy_update=2,
                 soft_tau=1e-3
                 ):

        """


        - gpu_id: 使用gpu id
        - num_inputs: 观测变量维度
        - num_actions: 控制维度
        - act_hidden_size: 策略网络隐层节点数
        - val_hidden_size: 评估网络隐层节点数
        - replay_buffer: 经验回放池对象
        - u_bounds: 控制指令上下界
        - exploration: 噪音探索器
        - batch_size: 每次训练从回放池选取的样本数
        - policy_lr: 策略网络学习率
        - value_lr: 评估网络学习率
        - noise_std: 为next_action添加的高斯噪音的方差
        - noise_clip: 噪音上下限
        - gamma: 累积奖赏的折扣因子
        - policy_update: 策略网络更新周期
        - soft_tau: 当前网络向target网络更新的速率

        :param gpu_id: 使用gpu id
        :param num_inputs: 观测变量维度
        :param num_actions: 控制维度
        :param act_hidden_size: 策略网络隐层节点数
        :param val_hidden_size: 评估网络隐层节点数
        :param replay_buffer: 经验回放池对象
        :param u_bounds: 控制指令上下界
        :param exploration: 噪音探索器
        :param batch_size: 每次训练从回放池选取的样本数
        :param policy_lr: 策略网络学习率
        :param value_lr: 评估网络学习率
        :param noise_std: 为next_action添加的高斯噪音的方差
        :param noise_clip: 噪音上下限
        :param gamma: 累积奖赏的折扣因子
        :param policy_update: 策略网络更新周期
        :param soft_tau: 当前网络向target网络更新的速率
        """
        super(Td3, self).__init__(gpu_id,
                                  replay_buffer=replay_buffer,
                                  u_bounds=u_bounds,
                                  exploration=exploration
                                  )
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.act_hidden_size = act_hidden_size
        self.val_hidden_size = val_hidden_size


        # current network
        self.policy_network = None
        self.value_network1 = None
        self.value_network2 = None


        # target network
        self.target_policy_network = None
        self.target_value_network1 = None
        self.target_value_network2 = None

        self.network_definition()


        # Learning parameters
        self.batch_size = batch_size
        self.value_loss = nn.MSELoss()
        self.policy_lr = policy_lr
        self.value_lr = value_lr

        self.value_optimizer1 = optim.Adam(self.value_network1.parameters(), lr=self.value_lr)
        self.value_optimizer2 = optim.Adam(self.value_network2.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.policy_lr)


        # RL parameters
        self.noise_std = noise_std
        self.gamma = gamma
        self.noise_clip = noise_clip
        self.policy_update = policy_update
        self.soft_tau = soft_tau

    def network_definition(self):
        self.value_network1 = ValueNetwork(self.num_inputs, self.num_actions, self.val_hidden_size, device=self.device)
        self.value_network2 = ValueNetwork(self.num_inputs, self.num_actions, self.val_hidden_size, device=self.device)
        self.policy_network = PolicyNetwork(self.num_inputs, self.num_actions, self.act_hidden_size, device=self.device)

        self.target_value_network1 = ValueNetwork(self.num_inputs, self.num_actions, self.val_hidden_size, device=self.device)
        self.target_value_network2 = ValueNetwork(self.num_inputs, self.num_actions, self.val_hidden_size, device=self.device)
        self.target_policy_network = PolicyNetwork(self.num_inputs, self.num_actions, self.act_hidden_size, device=self.device)

        copy_net(self.value_network1, self.target_value_network1)
        copy_net(self.value_network2, self.target_value_network2)
        copy_net(self.policy_network, self.target_policy_network)

    def policy_act(self, state):
        action = self.policy_network.get_action(state)
        return action

    def update_model(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.step)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        next_action = self.target_policy_network(next_state)
        noise = torch.normal(torch.zeros(next_action.size()), self.noise_std).to(self.device)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_action = next_action + noise

        target_q_value1 = self.target_value_network2(next_state, next_action)
        target_q_value2 = self.target_value_network2(next_state, next_action)

        # td3的核心,用两个target net的较小值来评估next_state
        target_q_value = torch.min(target_q_value1, target_q_value2)
        expect_q_value = reward + (1.0 - done)*target_q_value*self.gamma

        q_value1 = self.value_network1(state, action)
        q_value2 = self.value_network2(state, action)

        value_net1_loss = self.value_loss(expect_q_value.detach(), q_value1)
        value_net2_loss = self.value_loss(expect_q_value.detach(), q_value2)

        self.value_optimizer1.zero_grad()
        value_net1_loss.backward()
        self.value_optimizer1.step()

        self.value_optimizer2.zero_grad()
        value_net2_loss.backward()
        self.value_optimizer2.step()

        if self.step % self.policy_update:
            # 计算策略网络损失
            policy_loss = self.value_network1(state, self.policy_network(state))
            #
            policy_loss = - policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            soft_update(self.value_network1, self.target_value_network1, self.soft_tau)
            soft_update(self.value_network2, self.target_value_network2, self.soft_tau)
            soft_update(self.policy_network, self.target_policy_network, self.soft_tau)

    def _train(self, s, u, ns, r, done):
        if done is True:
            raise ValueError
        self.replay_buffer.push(s, u, ns, r, done)
        if len(self.replay_buffer) <= self.batch_size:
            return
        self.update_model()
