import random
import math
import csv


def compute_unbalance(masses, disk_unbalance, disk_angle, r=88.0):
    """
    计算给定排序下的合成不平衡量大小和角度。
    参数:
        masses: 按照安装顺序排列的叶片质量列表(长度40)
        disk_unbalance: 叶盘不平衡量 (标量, 单位 g·mm)
        disk_angle: 叶盘不平衡量的角度 (弧度制)
        r: 叶片中心半径, 默认 88mm
    返回:
        (U_mag, U_angle) -> (合成不平衡量大小, 合成不平衡量角度(弧度))
    """
    # 叶盘不平衡量的 x, y 分量
    Ux = disk_unbalance * math.cos(disk_angle)
    Uy = disk_unbalance * math.sin(disk_angle)

    # 计算每个安装位置的角度
    # i-th 位置对应角度 theta_i
    n = len(masses)
    for i, m in enumerate(masses):
        theta_i = 2.0 * math.pi * i / n
        # 不平衡分量 = mass_i * r
        Ux += m * r * math.cos(theta_i)
        Uy += m * r * math.sin(theta_i)

    # 合成不平衡量和相位
    U_mag = math.sqrt(Ux ** 2 + Uy ** 2)
    U_angle = math.atan2(Uy, Ux)
    return U_mag, U_angle


def main():
    """
    主函数：执行 2000 次模拟，每次随机生成 40 个叶片质量和叶盘不平衡量+角度，
    并通过 10000 次随机排序寻找最小合成不平衡量的排序方案，
    最后将结果写入 CSV 文件保存。
    """
    random.seed(42)  # 为了保证可重复性，可固定种子(可自由修改或去掉)

    # 打开CSV文件
    with open("rotor_data.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写表头：根据实际需求自行添加/修改
        writer.writerow([
            "iteration_index",  # 第几次实验
            "disk_unbalance(gmm)",  # 叶盘不平衡量
            "disk_angle(rad)",  # 叶盘不平衡角度(弧度)
            "blade_masses(g)",  # 40 个叶片质量
            "best_sequence",  # 最优排序(叶片下标的排列)
            "final_unbalance_magnitude",  # 最优不平衡量大小(g·mm)
            "final_unbalance_angle(rad)"  # 最优不平衡量角度(弧度)
        ])

        num_iterations = 20000
        num_blades = 40
        num_random_search = 10000

        for iteration in range(num_iterations):
            # 1) 随机生成 40 个叶片质量
            blade_masses = [random.uniform(39.9, 41.5) for _ in range(num_blades)]

            # 2) 随机生成叶盘不平衡量和角度
            disk_unb = random.uniform(0, 200)  # g·mm
            disk_ang = random.uniform(0, 2 * math.pi)  # rad

            # 3) 通过随机搜索找最优排序（不平衡量最小）
            # 为了记录最优解，需要保存：最优不平衡量、角度、排序
            best_unbalance_mag = float("inf")
            best_unbalance_angle = 0.0
            best_seq = None

            # 首先生成一个基础索引 [0, 1, 2, ..., 39]
            blade_indices = list(range(num_blades))

            for _ in range(num_random_search):
                # 打乱索引
                random.shuffle(blade_indices)
                # 根据当前索引顺序取得质量排列
                shuffled_masses = [blade_masses[i] for i in blade_indices]

                # 计算合成不平衡量
                U_mag, U_ang = compute_unbalance(shuffled_masses, disk_unb, disk_ang, r=88.0)

                # 如果当前不平衡量更优，则更新
                if U_mag < best_unbalance_mag:
                    best_unbalance_mag = U_mag
                    best_unbalance_angle = U_ang
                    best_seq = blade_indices[:]

            # 4) 将结果写入CSV
            # 注意：为保证 CSV 的可读性，一些列表字段可以转为字符串
            writer.writerow([
                iteration,
                f"{disk_unb:.4f}",
                f"{disk_ang:.6f}",
                # 叶片质量可以存为字符串，后面再解析
                ";".join(f"{m:.4f}" for m in blade_masses),
                ";".join(str(idx) for idx in best_seq),
                f"{best_unbalance_mag:.4f}",
                f"{best_unbalance_angle:.6f}"
            ])

            # 可以打印进度
            if (iteration + 1) % 100 == 0:
                print(f"Completed {iteration + 1}/{num_iterations} iterations.")


if __name__ == "__main__":
    main()
