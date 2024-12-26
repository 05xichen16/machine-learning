import csv


def load_rotor_data_from_csv(csv_path):
    """
    从 rotor_data.csv 中读取数据，并解析为特定形式的列表 data。
    返回: data (list of dict)
    """
    data = []
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 取出 CSV 中需要的字段
            disk_unbalance_str = row['disk_unbalance(gmm)']  # 叶盘不平衡量 (str)
            disk_angle_str = row['disk_angle(rad)']  # 叶盘不平衡角度 (str)

            blade_mass_str = row['blade_masses(g)']  # 用分号分割的质量列表 (str)
            best_sequence_str = row['best_sequence']  # 用分号分割的最优排序 (str)

            final_mag_str = row['final_unbalance_magnitude']  # 最优合成不平衡量 (str)
            final_angle_str = row['final_unbalance_angle(rad)']  # 最优合成不平衡角度 (str)

            # 将字符串转换成需要的类型
            rotor_unbalance_magnitude = float(disk_unbalance_str)
            rotor_unbalance_angle = float(disk_angle_str)

            # 将 blade_masses(g) 中的 40 个数值解析为 float 列表
            blade_mass_list = [float(x) for x in blade_mass_str.split(';')]

            # 最优排序序列 (下标 int)
            feasible_permutation = [int(x) for x in best_sequence_str.split(';')]

            result_unbalance_magnitude = float(final_mag_str)
            result_unbalance_angle = float(final_angle_str)

            # 组装为目标字典
            data_dict = {
                'rotor_unbalance_magnitude': rotor_unbalance_magnitude,
                'rotor_unbalance_angle': rotor_unbalance_angle,
                'blade_mass_list': blade_mass_list,
                'feasible_permutation': feasible_permutation,
                'result_unbalance_magnitude': result_unbalance_magnitude,
                'result_unbalance_angle': result_unbalance_angle
            }
            data.append(data_dict)
    return data


def split_dataset(data, train_size=16000, valid_size=2000, test_size=2000):
    """
    将 data 拆分为 train_data, valid_data, test_data
    假设总数据长度=2000(=1600+200+200)，如果实际长度不完全一致可自行调整。
    """
    # 这里简单地做切片，假设 data 已经按时间或某种顺序排好
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:train_size + valid_size + test_size]
    return train_data, valid_data, test_data


def main():
    csv_path = "rotor_data.csv"
    # 1. 读取 CSV 数据
    data = load_rotor_data_from_csv(csv_path)

    # 2. 划分数据集
    train_data, valid_data, test_data = split_dataset(data, 16000, 2000, 2000)

    print(f"Total data size: {len(data)}")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    # 可以检查一下读取解析后的数据格式
    # 这里仅示例打印前1条和最后1条
    print("Example of the first data item:", train_data[0])
    print("Example of the last data item:", test_data[-1])


if __name__ == "__main__":
    main()
