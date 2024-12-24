import torch.nn as nn


class CriticLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        初始化 Critic 网络。

        参数：
        - input_dim (int): 每个叶片的特征维度（仅质量）。
        - hidden_dim (int): LSTM 隐藏层维度。
        """
        super(CriticLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        前向传播。

        参数：
        - state (torch.Tensor): 输入状态，形状为 (batch_size, num_blades, input_dim)。

        返回：
        - state_value (torch.Tensor): 状态价值估计，形状为 (batch_size, 1)。
        """
        lstm_out, _ = self.lstm(state)  # [batch_size, num_blades, hidden_dim]
        # 使用最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        state_value = self.fc(last_out)  # [batch_size, 1]
        return state_value
