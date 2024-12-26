# import numpy as np
#
# num_blades = 4
# blade_masses = [1, 2, 3, 4]
# rotation_radius = 1
#
# rotor_unbalance_magnitude = 10
# rotor_unbalance_angle = np.pi
#
# rotor_vector = rotor_unbalance_magnitude * np.array([
#     np.cos(rotor_unbalance_angle),
#     np.sin(rotor_unbalance_angle)
# ])
#
# print(rotor_vector)
# def calculate_unbalance(arrangement):
#     """
#     计算给定叶片排列的总体不平衡量。
#
#     参数：
#     - arrangement (np.array): 当前叶片排列。
#
#     返回：
#     - unbalance (float): 总体不平衡量。
#     """
#     # 假设叶片均匀分布在360度内，每个叶片的角度为:
#     angles = np.linspace(0, 2 * np.pi, num_blades, endpoint=False)
#
#     # 计算叶片不平衡向量的总和
#     blades_vector = np.zeros(2)
#     for idx, blade in enumerate(arrangement):
#         mass = blade_masses[blade]
#         angle = angles[idx]
#         blades_vector += rotation_radius * mass * np.array([np.cos(angle), np.sin(angle)])
#         print(blades_vector)
#
#     # 总体不平衡向量 = 叶片不平衡向量 + 叶盘不平衡向量
#     total_vector = blades_vector + rotor_vector
#     print(total_vector)
#
#     # 总体不平衡量为不平衡向量的模
#     unbalance = np.linalg.norm(total_vector)
#     print(unbalance)
#     return unbalance
#
# calculate_unbalance([0, 1, 2, 3])
