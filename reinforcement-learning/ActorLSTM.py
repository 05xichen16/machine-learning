import torch.nn as nn


class ActorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        """
        初始化 Actor 网络。

        参数：
        - input_dim (int): 每个叶片的特征维度（仅质量）。
        - hidden_dim (int): LSTM 隐藏层维度。
        - num_actions (int): 动作空间大小。
        """
        super(ActorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        """
        前向传播。

        参数：
        - state (torch.Tensor): 输入状态，形状为 (batch_size, num_blades, input_dim)。

        返回：
        - action_probs (torch.Tensor): 动作概率分布，形状为 (batch_size, num_actions)。
        """
        lstm_out, _ = self.lstm(state)  # [batch_size, num_blades, hidden_dim]
        # 使用最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        action_scores = self.fc(last_out)  # [batch_size, num_actions]
        action_probs = nn.functional.softmax(action_scores, dim=-1)  # [batch_size, num_actions]
        return action_probs
