import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
import read_data

# ================== 3.1 引入依赖与数据集类 ==================

class RotorDataset(Dataset):
    """
    用于处理转子叶片排序数据的Dataset。
    假设传入的数据列表中，每条数据包含:
      - 'rotor_unbalance_magnitude': float
      - 'rotor_unbalance_angle': float
      - 'blade_mass_list': list(float), len=40
      - 'feasible_permutation': list(int), len=40
    """
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # 1) 读取叶片质量
        blade_masses = item['blade_mass_list']  # list of 40 floats

        # 2) 读取叶盘不平衡量 & 角度
        rotor_mag = item['rotor_unbalance_magnitude']
        rotor_ang = item['rotor_unbalance_angle']

        # 3) 构建输入特征 x: shape [40, 3]
        #    [mass, rotor_mag_cos, rotor_mag_sin]
        #    这样做可以更有效地编码叶盘不平衡向量
        x = np.zeros((40, 3), dtype=np.float32)
        for i in range(40):
            x[i, 0] = blade_masses[i]
            x[i, 1] = rotor_mag * math.cos(rotor_ang)
            x[i, 2] = rotor_mag * math.sin(rotor_ang)

        # 4) 可行解排序
        perm = np.array(item['feasible_permutation'], dtype=np.int64)  # [40]

        return {
            'x': x,       # [40, 3]
            'perm': perm  # [40]
        }

def collate_fn(batch):
    """
    将一个batch的样本打包处理为 (xs, perms)
    xs: [B, 40, 3]
    perms: [B, 40]
    """
    xs = []
    perms = []
    for d in batch:
        xs.append(d['x'])
        perms.append(d['perm'])
    xs = np.stack(xs, axis=0)       # [B, 40, 3]
    perms = np.stack(perms, axis=0) # [B, 40]

    xs = torch.from_numpy(xs)
    perms = torch.from_numpy(perms)
    return xs, perms

# ================== 3.2 模型结构：Encoder、Decoder、PointerNet ==================

class Encoder(nn.Module):
    """
    一个双向 LSTM 编码器，将输入 [B, N, D] -> [B, N, H]
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.proj = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, x):
        # x: [B, N, input_dim]
        out, (h, c) = self.lstm(x)  # out: [B, N, 2*H]
        out = self.proj(out)        # [B, N, H]
        return out, (h, c)


class Decoder(nn.Module):
    """
    Pointer Network 解码器，带掩码机制。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)

        # 注意力相关层
        self.W_ref = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v     = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, target_permutation=None):
        """
        encoder_outputs: [B, N, H]
        target_permutation: [B, N] (optional, used for teacher forcing)
        返回 pointer_logits: [B, N, N]
        """
        B, N, H = encoder_outputs.size()

        # 初始化隐藏状态和细胞状态
        hidden = torch.zeros(B, self.hidden_dim, device=encoder_outputs.device)
        cell   = torch.zeros(B, self.hidden_dim, device=encoder_outputs.device)

        # 预计算 encoder_outputs_proj
        encoder_outputs_proj = self.W_ref(encoder_outputs)  # [B, N, H]

        pointer_logits = []

        # 创建一个mask tensor，初始时所有位置都可选
        mask = torch.ones(B, N, device=encoder_outputs.device)  # [B, N]

        for t in range(N):
            if target_permutation is not None:
                # Teacher Forcing: 使用目标排序中的当前叶片作为输入
                # 获取第 t 步的目标叶片索引
                target = target_permutation[:, t]  # [B]
                # 将目标叶片的特征作为 LSTMCell 的输入
                # Gather encoder_outputs at target indices
                target_idx = target.unsqueeze(1).unsqueeze(2).expand(-1, 1, H)  # [B, 1, H]
                target_features = torch.gather(encoder_outputs, 1, target_idx).squeeze(1)  # [B, H]
                hidden, cell = self.lstm_cell(target_features, (hidden, cell))
            else:
                # 在推断时，使用上一步的隐藏状态作为输入（可以采用不同策略）
                # 这里简化为使用当前隐藏状态
                hidden, cell = self.lstm_cell(hidden, (hidden, cell))

            # 计算注意力
            query = self.W_q(hidden)                # [B, H]
            query = query.unsqueeze(1).expand(-1, N, -1)  # [B, N, H]
            sum_  = torch.tanh(encoder_outputs_proj + query)  # [B, N, H]
            logits = self.v(sum_).squeeze(-1)       # [B, N]

            # 应用掩码，将已选中的叶片的logits设为一个非常小的值
            logits = logits.masked_fill(mask == 0, -1e9)  # [B, N]

            pointer_logits.append(logits)

            if target_permutation is not None:
                # 在训练时，根据目标排序更新mask
                selected = target[:, t]  # [B]
                mask = mask.masked_fill(torch.arange(B).to(encoder_outputs.device).unsqueeze(1) == selected.unsqueeze(1), 0)

        # pointer_logits: list of length N, each [B, N]
        pointer_logits = torch.stack(pointer_logits, dim=1)  # [B, N, N]
        return pointer_logits

class PointerNet(nn.Module):
    """
    Pointer Network 将 Encoder 和 Decoder 整合在一起
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim)

    def forward(self, x, target_permutation=None):
        """
        x: [B, N, input_dim]
        target_permutation: [B, N] (optional, used for teacher forcing)
        返回 pointer_logits: [B, N, N]
        """
        encoder_out, _ = self.encoder(x)
        pointer_logits = self.decoder(encoder_out, target_permutation)
        return pointer_logits

# ================== 3.3 损失函数 ==================

def pointer_network_loss(pointer_logits, target_permutation):
    """
    pointer_logits: [B, N, N]
      - 对第 b 个样本, 第 t 步, 第 i 个输入元素的 logits => pointer_logits[b,t,i]
    target_permutation: [B, N], each in [0..N-1]
      - 对第 b 个样本, 第 t 步, 真实应选择的输入元素索引
    """
    B, N, _ = pointer_logits.size()
    # 变形后用 CrossEntropyLoss
    logits_2d = pointer_logits.view(B*N, N)         # [B*N, N]
    target_1d = target_permutation.view(B*N)        # [B*N]

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits_2d, target_1d)
    return loss

# ================== 3.4 解码函数（带掩码的贪心解码） ==================

def greedy_decode(model, x, device='cuda'):
    """
    x: [1, N, input_dim]
    返回 final_seq: list of int, 表示一个排列
    """
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        pointer_logits = model(x)      # [1, N, N]
        pointer_probs = torch.softmax(pointer_logits, dim=-1)  # [1, N, N]

    # 创建一个mask tensor，初始时所有位置都可选
    B, N, _ = pointer_probs.size()
    mask = torch.ones(B, N, device=x.device)  # [B, N]

    permutation = []
    for t in range(N):
        logits = pointer_logits[:, t, :]  # [B, N]
        logits = logits.masked_fill(mask == 0, -1e9)  # Apply mask
        probs = torch.softmax(logits, dim=-1)      # [B, N]
        selected = torch.argmax(probs, dim=-1)      # [B]
        permutation.append(selected.cpu().numpy())

        # 更新mask
        mask[torch.arange(B), selected] = 0

    # 由于B=1，提取第一维
    final_seq = permutation[0].tolist()
    return final_seq

# ================== 3.5 训练与验证循环 ==================

def train_pointer_net(model, train_loader, valid_loader,
                      device='cuda', lr=1e-3, num_epochs=20):
    """
    训练指针网络
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0
        count = 0
        for xs, perms in train_loader:
            xs = xs.to(device)      # [B, 40, 3]
            perms = perms.to(device) # [B, 40]
            print("perms shape: ", perms.shape)
            B = xs.size(0)

            optimizer.zero_grad()
            pointer_logits = model(xs, target_permutation=perms)  # [B, 40, 40]
            loss = pointer_network_loss(pointer_logits, perms)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            count += B

        avg_train_loss = total_loss / count
        train_losses.append(avg_train_loss)

        # ---------- 验证 ----------
        model.eval()
        total_val_loss = 0
        count_val = 0
        with torch.no_grad():
            for xs, perms in valid_loader:
                xs = xs.to(device)
                perms = perms.to(device)
                Bv = xs.size(0)
                pointer_logits = model(xs, target_permutation=perms)  # [B, 40, 40]
                loss_v = pointer_network_loss(pointer_logits, perms)
                total_val_loss += loss_v.item() * Bv
                count_val += Bv
        avg_val_loss = total_val_loss / count_val
        valid_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Valid Loss: {avg_val_loss:.4f}")

    return train_losses, valid_losses

# ================== 3.6 主程序示例 ==================

def main():
    # ================== 数据加载 ==================
    # 假设您的数据已经被加载为 train_data, valid_data, test_data
    # 这里假设您已经有这三个变量
    # 例如：
    # train_data = [...]
    # valid_data = [...]
    # test_data  = [...]
    # 您需要根据实际情况加载数据

    csv_path = "rotor_data.csv"
    data = read_data.load_rotor_data_from_csv(csv_path)

    # 2. 划分数据集
    train_data, valid_data, test_data = read_data.split_dataset(data, 16000, 2000, 2000)

    # 创建Dataset和DataLoader
    train_dataset = RotorDataset(train_data)
    valid_dataset = RotorDataset(valid_data)
    test_dataset  = RotorDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # ================== 模型初始化 ==================
    input_dim = 3   # [mass, rotor_mag_cos, rotor_mag_sin]
    hidden_dim = 128
    model = PointerNet(input_dim, hidden_dim)

    # ================== 训练 ==================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20
    learning_rate = 1e-3

    train_losses, valid_losses = train_pointer_net(model, train_loader, valid_loader,
                                                  device=device, lr=learning_rate, num_epochs=num_epochs)

    # ================== 绘制训练曲线 ==================
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Over Epochs')

    plt.tight_layout()
    plt.show()

    # ================== 测试与推断 ==================
    # 定义一个函数来计算转子不平衡量（与您的计算方式一致）
    def calculate_unbalance(blade_masses, permutation, rotor_mag_cos, rotor_mag_sin, rotation_radius=0.5):
        """
        计算转子不平衡量。
        blade_masses: list of 40 floats
        permutation: list of 40 ints
        rotor_mag_cos: float
        rotor_mag_sin: float
        rotation_radius: float
        返回: float
        """
        imbalance_vector = np.zeros(2)
        angles = np.linspace(0, 2*np.pi, 40, endpoint=False)
        for idx, blade_idx in enumerate(permutation):
            mass = blade_masses[blade_idx]
            angle = angles[idx]
            imbalance_vector += rotation_radius * mass * np.array([np.cos(angle), np.sin(angle)])
        # 添加叶盘的不平衡量
        imbalance_vector += np.array([rotor_mag_cos, rotor_mag_sin])
        unbalance = np.linalg.norm(imbalance_vector)
        return unbalance

    # 测试模型
    model.eval()
    total_unbalance = 0
    count = 0
    for xs, perms in test_loader:
        xs = xs.to(device)      # [B, 40, 3]
        perms = perms.numpy()    # [B, 40]
        B = xs.size(0)
        for b in range(B):
            x = xs[b].unsqueeze(0)  # [1, 40, 3]
            pred_perm = greedy_decode(model, x, device=device)  # list of 40 ints
            blade_masses = x[0,:,0].cpu().numpy().tolist()
            rotor_mag_cos = x[0,0,1].item()
            rotor_mag_sin = x[0,0,2].item()
            unbalance = calculate_unbalance(blade_masses, pred_perm, rotor_mag_cos, rotor_mag_sin)
            total_unbalance += unbalance
            count += 1

    avg_unbalance = total_unbalance / count
    print(f"Average Unbalance on Test Set: {avg_unbalance:.4f}")

    # ================== 可视化部分 ==================
    # 选择一个测试样本，展示排序结果
    sample_idx = 0
    sample = test_data[sample_idx]
    x_input = np.zeros((1, 40, 3), dtype=np.float32)
    for i in range(40):
        x_input[0, i, 0] = sample['blade_mass_list'][i]
        rotor_mag = sample['rotor_unbalance_magnitude']
        rotor_ang = sample['rotor_unbalance_angle']
        x_input[0, i, 1] = rotor_mag * math.cos(rotor_ang)
        x_input[0, i, 2] = rotor_mag * math.sin(rotor_ang)

    x_tensor = torch.from_numpy(x_input).to(device)
    predicted_perm = greedy_decode(model, x_tensor, device=device)
    ground_truth_perm = sample['feasible_permutation']

    print(f"Sample Index: {sample_idx}")
    print(f"Predicted Permutation: {predicted_perm}")
    print(f"Ground Truth Permutation: {ground_truth_perm}")

    # 计算预测排序的不平衡量与真实排序的不平衡量
    predicted_unbalance = calculate_unbalance(sample['blade_mass_list'], predicted_perm,
                                             sample['rotor_unbalance_magnitude']*math.cos(sample['rotor_unbalance_angle']),
                                             sample['rotor_unbalance_magnitude']*math.sin(sample['rotor_unbalance_angle']))
    ground_truth_unbalance = sample['result_unbalance_magnitude']

    print(f"Predicted Unbalance: {predicted_unbalance:.4f}")
    print(f"Ground Truth Unbalance: {ground_truth_unbalance:.4f}")

if __name__ == "__main__":
    main()

