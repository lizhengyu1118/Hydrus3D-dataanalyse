import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- 配置参数 ---
# 您的观测点坐标映射文件路径，确保这个文件存在且路径正确
OBS_POINT_MAPPING_FILE =  'obs_point_mapping.csv'

# --- 函数：加载观测点映射文件 ---
def load_obs_point_mapping(mapping_filepath):
    """
    加载观测点ID到物理坐标的映射文件。
    要求包含 PhysicalX, PhysicalY, PhysicalZ 列。
    返回一个DataFrame。
    """
    try:
        mapping_df = pd.read_csv(mapping_filepath)
        # 验证关键列是否存在
        required_cols = ['Profile', 'Group', 'ObsPointID', 'PhysicalX', 'PhysicalY', 'PhysicalZ']
        if not all(col in mapping_df.columns for col in required_cols):
            # PhysicalColumn 和 PhysicalDepth 对于3D验证不是必需的，但对于后续2D可视化需要
            # 这里仅检查3D验证需要的列，但建议映射文件包含所有列
            print("警告: 映射文件中未找到 PhysicalColumn 或 PhysicalDepth 列，不影响3D验证，但后续可能需要。")
            required_cols_3d_only = ['Profile', 'Group', 'ObsPointID', 'PhysicalX', 'PhysicalY', 'PhysicalZ']
            if not all(col in mapping_df.columns for col in required_cols_3d_only):
                 raise ValueError(f"映射文件必须包含列: {required_cols_3d_only}")

        print(f"成功加载观测点映射文件: {mapping_filepath}")
        return mapping_df
    except FileNotFoundError:
        print(f"错误: 观测点映射文件未找到在 {mapping_filepath}")
        print("请确保文件存在且路径正确，并且包含 Profile, Group, ObsPointID, PhysicalX, PhysicalY, PhysicalZ 列。")
        return None
    except Exception as e:
        print(f"加载映射文件时出错: {e}")
        return None

# --- 函数：可视化映射文件中的点 (三维验证) ---
def verify_mapping_file_3d(mapping_df):
    """
    可视化观测点映射文件中定义的所有点在三维空间中的位置。
    """
    if mapping_df is None or mapping_df.empty:
        print("映射文件数据为空，无法进行验证可视化。")
        return

    print(f"\n映射文件总点数: {len(mapping_df)}")

    x = mapping_df['PhysicalX']
    y = mapping_df['PhysicalY']
    z = mapping_df['PhysicalZ']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 可以按剖面或分组给点上色以便区分
    # 例如按剖面上色，并添加图例
    unique_profiles = mapping_df['Profile'].unique()
    # 使用一个颜色映射，确保颜色数量足够多或使用循环颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_profiles))) # tab20提供20种颜色
    profile_color_map = dict(zip(unique_profiles, colors))

    for profile in unique_profiles:
        profile_df = mapping_df[mapping_df['Profile'] == profile]
        if not profile_df.empty:
            ax.scatter(profile_df['PhysicalX'], profile_df['PhysicalY'], profile_df['PhysicalZ'],
                       color=profile_color_map[profile], label=f'Profile {profile}', s=50) # s控制点的大小

    # 如果点数不多，可以考虑添加文本标签 (可能会非常拥挤，慎用)
    # limited_df = mapping_df.sample(min(len(mapping_df), 50)) # 只标注一部分点，避免拥挤
    # for index, row in limited_df.iterrows():
    #    ax.text(row['PhysicalX'], row['PhysicalY'], row['PhysicalZ'],
    #            f"{row['Profile']}-{row['Group']}-{row['ObsPointID']}", size=8, zorder=1, color='k')


    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Visualization of Observation Points from Mapping File')

    # ax.invert_zaxis() # 如果需要反转Z轴，使深度向下增加

    ax.legend() # 显示剖面图例
    plt.tight_layout()
    plt.show()


# --- 主执行流程 (独立运行) ---
if __name__ == "__main__":
    print("--- 验证观测点映射文件 (3D可视化) ---")
    # 1. 加载观测点映射文件
    obs_mapping_df = load_obs_point_mapping(OBS_POINT_MAPPING_FILE)

    # 2. 进行三维可视化验证
    if obs_mapping_df is not None:
        verify_mapping_file_3d(obs_mapping_df)

    print("验证程序结束。")