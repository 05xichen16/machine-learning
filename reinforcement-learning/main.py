import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from RotorBladeSortingEnv import RotorBladeSortingEnv
from ActorLSTM import ActorLSTM
from CriticLSTM import CriticLSTM

# num_blades = 40  # 叶片数量
#
# # 定义每个叶片的质量
# blade_masses = mass_matrix = [40.953, 40.269, 40.985, 41.197, 40.514, 40.414, 40.356, 39.943, 41.077,
#                               41.164, 40.861, 40.455, 40.975, 40.055, 40.13, 40.818, 41.153, 40.463,
#                               40.907, 40.496, 40.949, 39.994, 40.494, 41.11, 40.636, 41.168, 40.023,
#                               40.427, 40.089, 40.115, 40.726, 40.535, 40.69, 41.009, 39.9, 41.045,
#                               40.9, 39.815, 41.137, 41.019]
#
# # 定义转子的旋转半径（所有叶片一致）
# rotation_radius = 88
#
# # 定义叶盘的不平衡量大小和角度
# rotor_unbalance_magnitude = 100  # 例如，单位与质量×半径一致
# rotor_unbalance_angle = np.pi / 4  # 45度，单位为弧度
num_blades = 10
blade_masses = [1.0, 1.2, 0.9, 1.1, 1.0, 1.3, 0.95, 1.05, 1.15, 1.0]
rotation_radius = 1
rotor_unbalance_magnitude = 0
rotor_unbalance_angle = 0

env = RotorBladeSortingEnv(num_blades, blade_masses, rotation_radius, rotor_unbalance_magnitude, rotor_unbalance_angle)

# 超参数
input_dim = 1  # 每个叶片的特征维度（仅质量）
hidden_dim = 128
num_actions = env.action_space.n
learning_rate = 1e-3
gamma = 0.99  # 折扣因子
num_episodes = 10000

# 实例化 Actor 和 Critic 网络
actor = ActorLSTM(input_dim, hidden_dim, num_actions)
critic = CriticLSTM(input_dim, hidden_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor.to(device)
critic.to(device)

# 定义优化器
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

# 定义损失函数
mse_loss = nn.MSELoss()


def select_action(state):
    """
    根据当前状态选择动作。

    参数：
    - state (np.array): 当前状态，形状为 (num_blades,)。

    返回：
    - action (int): 选择的动作索引。
    - log_prob (torch.Tensor): 动作的对数概率。
    """
    state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(-1).to(device)  # [1, num_blades, 1]
    action_probs = actor(state)  # [1, num_actions]
    action_dist = torch.distributions.Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    return action.item(), log_prob


# 记录训练过程
rewards = []
unbalances = []

for episode in range(num_episodes):
    state = env.reset()  # [num_blades,]
    done = False
    total_reward = 0.0
    while not done:
        # 选择动作
        action, log_prob = select_action(state)

        # 执行动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # 转换为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(-1).to(device)  # [1, num_blades, 1]
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(-1).to(device)  # [1, num_blades, 1]
        reward_tensor = torch.FloatTensor([[reward]]).to(device)  # [1, 1]

        # 计算 Critic 估计
        value = critic(state_tensor)  # [1, 1]
        next_value = critic(next_state_tensor)  # [1, 1]

        # 计算 TD 目标和 TD 误差
        if done:
            td_target = reward_tensor
        else:
            td_target = reward_tensor + gamma * next_value.detach()
        td_error = td_target - value  # [1, 1]

        # 更新 Critic
        critic_loss = mse_loss(value, td_target)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # 更新 Actor
        advantage = td_error.detach()
        actor_loss = -log_prob * advantage
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 更新状态
        state = next_state

    # 记录奖励和不平衡量
    rewards.append(total_reward)
    unbalances.append(info['unbalance'])

    # 打印训练信息
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.4f}, Final Unbalance: {info['unbalance']:.4f}")

# # 绘制训练曲线
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(rewards)
# plt.title('Total Reward per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
#
# plt.subplot(1, 2, 2)
# plt.plot(unbalances)
# plt.title('Final Unbalance per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Final Unbalance')
#
# plt.tight_layout()
# plt.show()
