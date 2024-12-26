import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# PPO Actor-Critic 模型
class PPOActorCritic(nn.Module):
    def __init__(self, num_blades, feature_dim, latent_dim=128):
        super(PPOActorCritic, self).__init__()
        # 编码器
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=latent_dim, batch_first=True)
        # 策略网络 (Actor)
        self.policy = nn.Linear(latent_dim, num_blades)
        # 值网络 (Critic)
        self.value = nn.Linear(latent_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后时间步的输出
        policy = torch.softmax(self.policy(lstm_out), dim=-1)
        value = self.value(lstm_out)
        return policy, value

# 超参数
num_steps = 100
num_blades = 67
feature_dim = 2
latent_dim = 128
batch_size = 16
clip_ratio = 0.2
gamma = 0.99
lam = 0.95
learning_rate = 1e-3

# PPO 训练逻辑
class PPOTrainer:
    def __init__(self, model, optimizer, clip_ratio, gamma, lam):
        self.model = model
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (1 - dones[t]) * self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return np.array(advantages, dtype=np.float32)

    def train(self, states, actions, rewards, values, dones):
        # 计算优势函数
        advantages = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # 前向传播
        old_policies, old_values = self.model(states)
        old_log_probs = torch.log(old_policies.gather(1, actions.unsqueeze(1)).squeeze())

        for _ in range(4):  # 多次更新提高收敛性
            policies, values = self.model(states)
            log_probs = torch.log(policies.gather(1, actions.unsqueeze(1)).squeeze())

            # 策略损失
            ratios = torch.exp(log_probs - old_log_probs)
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            # 值损失
            value_loss = ((rewards - values.squeeze()) ** 2).mean()

            # 总损失
            loss = policy_loss + 0.5 * value_loss

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 模型实例化
model = PPOActorCritic(num_blades, feature_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
trainer = PPOTrainer(model, optimizer, clip_ratio, gamma, lam)

# 生成模拟数据
def generate_simulation_data(batch_size, num_blades, feature_dim):
    states = np.random.rand(batch_size, num_blades, feature_dim)  # 随机生成叶片特征
    actions = np.random.randint(0, num_blades, size=(batch_size,))  # 随机选择动作
    rewards = -np.sum(np.abs(states[:, :, 0] - 0.5), axis=1)  # 奖励函数（模拟）
    dones = np.zeros(batch_size, dtype=np.float32)  # 假设未终止
    return states, actions, rewards, dones

# 训练循环
for step in range(num_steps):
    # 生成训练数据
    states, actions, rewards, dones = generate_simulation_data(batch_size, num_blades, feature_dim)

    # 前向传播获取值函数
    with torch.no_grad():
        _, values = model(torch.tensor(states, dtype=torch.float32))
        values = values.squeeze().numpy()

    # 更新模型
    trainer.train(states, actions, rewards, values, dones)

    if step % 100 == 0:
        print(f"Step {step}, Average Reward: {np.mean(rewards)}")

# 测试排序结果
def evaluate_model(model, test_states):
    with torch.no_grad():
        policies, _ = model(torch.tensor(test_states, dtype=torch.float32))
        sorted_indices = torch.argmax(policies, dim=-1)
    return sorted_indices.numpy()

# 测试数据
test_states = np.random.rand(10, num_blades, feature_dim)
test_sorted_indices = evaluate_model(model, test_states)
print("Test Sorted Indices:", test_sorted_indices)
