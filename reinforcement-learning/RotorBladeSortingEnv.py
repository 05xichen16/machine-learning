import gym
from gym import spaces
import numpy as np


class RotorBladeSortingEnv(gym.Env):
    def __init__(self, num_blades, blade_masses, rotation_radius, rotor_unbalance_magnitude, rotor_unbalance_angle):
        """
        初始化环境。

        参数：
        - num_blades (int): 叶片数量。
        - blade_masses (list或np.array): 每个叶片的质量。
        - rotation_radius (float): 转子的旋转半径（所有叶片一致）。
        - rotor_unbalance_magnitude (float): 叶盘的不平衡量大小。
        - rotor_unbalance_angle (float): 叶盘不平衡量的角度（弧度）。
        """
        super(RotorBladeSortingEnv, self).__init__()
        self.num_blades = num_blades
        self.blade_masses = np.array(blade_masses)
        self.rotation_radius = rotation_radius
        self.rotor_unbalance_magnitude = rotor_unbalance_magnitude
        self.rotor_unbalance_angle = rotor_unbalance_angle

        # 验证输入
        assert len(self.blade_masses) == num_blades, "blade_masses长度应与num_blades一致"

        # 动作空间：所有可能的两两交换，共n(n-1)/2个动作
        self.action_space = spaces.Discrete(num_blades * (num_blades - 1) // 2)

        # 状态空间：叶片的排列顺序，仅包含质量
        self.observation_space = spaces.Box(low=0, high=np.max(self.blade_masses),
                                            shape=(num_blades,), dtype=np.float32)

        # 最大步骤数
        self.max_steps = num_blades * 2
        self.current_step = 0

        # 初始化叶片排列（随机排列）
        self.initial_arrangement = np.arange(num_blades)
        np.random.shuffle(self.initial_arrangement)
        self.state = self.get_state(self.initial_arrangement)

        # 预计算叶盘的不平衡向量
        self.rotor_vector = self.rotor_unbalance_magnitude * np.array([
            np.cos(self.rotor_unbalance_angle),
            np.sin(self.rotor_unbalance_angle)
        ])

    def get_state(self, arrangement):
        """
        将叶片排列转换为状态表示，仅包含质量。

        参数：
        - arrangement (np.array): 当前叶片排列。

        返回：
        - state (np.array): 状态表示，形状为(num_blades,)。
        """
        return self.blade_masses[arrangement]

    def reset(self, **kwargs):
        """
        重置环境到初始状态。

        返回：
        - state (np.array): 初始状态。
        """
        self.initial_arrangement = np.arange(self.num_blades)
        np.random.shuffle(self.initial_arrangement)
        self.state = self.get_state(self.initial_arrangement)
        self.current_step = 0
        return self.state

    def step(self, action):
        """
        执行动作，返回下一个状态、奖励、是否结束和额外信息。

        参数：
        - action (int): 动作索引，表示交换的叶片对。

        返回：
        - next_state (np.array): 下一个状态。
        - reward (float): 奖励。
        - done (bool): 是否结束。
        - info (dict): 额外信息。
        """
        # 将动作索引映射到叶片对
        pair = self.action_index_to_pair(action)
        # 执行动作：交换叶片
        new_arrangement = self.initial_arrangement.copy()
        new_arrangement[pair[0]], new_arrangement[pair[1]] = new_arrangement[pair[1]], new_arrangement[pair[0]]

        # 更新当前排列
        self.initial_arrangement = new_arrangement
        self.state = self.get_state(new_arrangement)
        self.current_step += 1

        # 计算不平衡量
        unbalance = self.calculate_unbalance(new_arrangement)

        # 定义奖励为不平衡量的负值
        reward = -unbalance

        # 定义终止条件
        done = self.current_step >= self.max_steps or unbalance <= 1e-3  # 设定阈值

        info = {'unbalance': unbalance}

        return self.state, reward, done, info

    def action_index_to_pair(self, action):
        """
        将动作索引映射到叶片对 (i, j)。

        参数：
        - action (int): 动作索引。

        返回：
        - pair (tuple): 叶片对 (i, j)。
        """
        idx = 0
        for i in range(self.num_blades):
            for j in range(i + 1, self.num_blades):
                if idx == action:
                    return i, j
                idx += 1
        return 0, 1  # 默认交换对

    def calculate_unbalance(self, arrangement):
        """
        计算给定叶片排列的总体不平衡量。

        参数：
        - arrangement (np.array): 当前叶片排列。

        返回：
        - unbalance (float): 总体不平衡量。
        """
        # 假设叶片均匀分布在360度内，每个叶片的角度为:
        angles = np.linspace(0, 2 * np.pi, self.num_blades, endpoint=False)

        # 计算叶片不平衡向量的总和
        blades_vector = np.zeros(2)
        for idx, blade in enumerate(arrangement):
            mass = self.blade_masses[blade]
            angle = angles[idx]
            blades_vector += self.rotation_radius * mass * np.array([np.cos(angle), np.sin(angle)])

        # 总体不平衡向量 = 叶片不平衡向量 + 叶盘不平衡向量
        total_vector = blades_vector + self.rotor_vector

        # 总体不平衡量为不平衡向量的模
        unbalance = np.linalg.norm(total_vector)
        return unbalance
