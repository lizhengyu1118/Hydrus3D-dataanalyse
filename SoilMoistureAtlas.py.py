import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from mpl_toolkits.mplot3d import Axes3D

# --- 配置参数 ---
PROJECT_ROOT_DIR = './Hydrus3D_Simulations' # 您的项目根目录路径
OBS_POINT_MAPPING_FILE = 'obs_point_mapping.csv' # 观测点坐标映射文件路径
PROFILE_LOCATIONS = ['05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55'] # 您的剖面位置列表
OBS_GROUPS = ['A', 'B', 'C', 'D'] # 您的观测点分组列表
LAND_USE_TYPES = ['exoticgrass', 'naturalgrass', 'exoticshrub', 'arablecrop'] # 您的土地利用类型列表
OBS_OUT_FILENAME = 'Obs.out' # Hydrus观测点输出文件名
TIME_COLUMN_NAME = 'Time' # Obs.out 文件中的时间列名
TARGET_VAR = 'theta' # 我们要提取的变量名称

# --- 函数：加载观测点映射文件 ---
def load_obs_point_mapping(mapping_filepath):
    """
    加载观测点ID到物理坐标的映射文件。
    现在要求包含 PhysicalX, PhysicalY, PhysicalZ 列。
    返回一个DataFrame。
    """
    try:
        mapping_df = pd.read_csv(mapping_filepath)
        # 验证关键列是否存在 (新增 3D 坐标列检查)
        required_cols = ['Profile', 'Group', 'ObsPointID', 'PhysicalColumn', 'PhysicalDepth', 'PhysicalX', 'PhysicalY', 'PhysicalZ']
        if not all(col in mapping_df.columns for col in required_cols):
            raise ValueError(f"映射文件必须包含列: {required_cols}")
        print(f"成功加载观测点映射文件: {mapping_filepath}")
        return mapping_df
    except FileNotFoundError:
        print(f"错误: 观测点映射文件未找到在 {mapping_filepath}")
        print("请确保文件存在且路径正确，并且包含 PhysicalX, PhysicalY, PhysicalZ 列。")
        return None
    except Exception as e:
        print(f"加载映射文件时出错: {e}")
        return None

# --- 函数：读取单个Obs.out文件，提取指定变量列 ---
def read_obs_out(filepath, target_variable=TARGET_VAR):
    """
    读取Hydrus的Obs.out文件，跳过注释行和Node行，提取指定变量列。
    并按顺序将变量列重命名为 Point1, Point2, ...
    """
    try:
        # 读取所有行
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 找到第一个非注释、非空行 (通常是Node行)
        node_line_index = -1
        for i, line in enumerate(lines):
            if not line.strip().startswith('#') and line.strip():
                node_line_index = i
                break

        if node_line_index == -1:
            print(f"警告: 文件 {filepath} 中未找到数据行（无非注释非空行）。")
            return None

        # 实际的表头行是 Node 行之后的第一个非注释、非空行
        header_line_index = -1
        for i in range(node_line_index + 1, len(lines)):
            if not lines[i].strip().startswith('#') and lines[i].strip():
                header_line_index = i
                break

        if header_line_index == -1:
            print(f"警告: 文件 {filepath} 在 Node 行后未找到表头行。")
            return None

        # 使用pandas读取，跳过 Node 行及其之前的注释行
        # skiprows 需要跳过 header_line_index 之前的所有行
        df = pd.read_csv(filepath, sep='\s+', skiprows=header_line_index, header=0, comment='#')

        # 移除全NaN的列（有时Obs.out末尾会有）
        df = df.dropna(axis=1, how='all')

        # 验证是否成功读取以及是否包含Time列
        if TIME_COLUMN_NAME not in df.columns or df.shape[1] < 2:
             print(f"警告: 文件 {filepath} 读取后格式异常，可能不是标准的Obs.out格式或没有观测点数据。")
             print("尝试直接读取并查看前几行:")
             try:
                 df_test = pd.read_csv(filepath, sep='\s+', comment='#', header=None)
                 print(df_test.head())
             except Exception as e_test:
                 print(f"直接读取也失败: {e_test}")
             return None

        # 提取时间列和目标变量列
        time_col_data = df[TIME_COLUMN_NAME]
        target_cols_data = df.filter(like=target_variable) # 查找所有包含 target_variable 的列

        # 验证是否找到了目标变量列
        if target_cols_data.empty:
            print(f"警告: 在文件 {filepath} 中未找到 '{target_variable}' 列。")
            return None

        # 重命名目标变量列为 Point1, Point2, ... 根据它们在文件中的顺序
        # 这里假设 Hydrus 按照观测点设置的顺序输出列
        new_point_col_names = [f'Point{i+1}' for i in range(target_cols_data.shape[1])]
        target_cols_data.columns = new_point_col_names

        # 将时间列和重命名后的目标变量列合并
        processed_df = pd.concat([time_col_data, target_cols_data], axis=1)

        # print(f"  - 成功读取并提取 '{target_variable}' 数据从 {filepath}")
        # print(f"    提取列: {processed_df.columns.tolist()}")

        return processed_df

    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"文件为空或只有注释行: {filepath}")
        return None
    except Exception as e:
        print(f"读取文件 {filepath} 时出错: {e}")
        # traceback.print_exc() # 打印详细错误信息，调试时开启
        return None


# --- 函数：合并一个剖面的所有分组数据 ---
def merge_profile_data(base_dir, profile, land_use, obs_groups, mapping_df):
    """
    合并指定土地利用类型和剖面位置下的所有分组(A, B, C, D)的Obs.out数据。
    返回一个包含所有观测点时间序列的DataFrame。
    同时筛选出时间为整数天的数据。
    """
    merged_df = None

    for group in obs_groups:
        # 构建可能的文件夹名称模式 (日期-剖面-分组-土地利用)
        search_pattern = f"*-{profile}-{group}-{land_use}"
        matching_dirs = [d for d in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, d)) and re.match(r'\d{6,8}-' + re.escape(profile) + '-' + re.escape(group) + '-' + re.escape(land_use) + r'$', d)]

        if not matching_dirs:
            # print(f"警告: 未找到匹配的文件夹模式 {search_pattern} 在 {base_dir}。跳过该分组。") # 避免过多警告
            continue

        # 理论上应该只有一个匹配的文件夹，选择第一个
        sim_folder = matching_dirs[0]
        obs_filepath = os.path.join(base_dir, sim_folder, OBS_OUT_FILENAME)

        if not os.path.exists(obs_filepath):
            # print(f"警告: 文件 {obs_filepath} 不存在。跳过该分组。") # 避免过多警告
            continue

        print(f"  正在读取: {obs_filepath}") # 保留读取进度提示
        # read_obs_out 现在只返回 Time 和 PointX 列 (PointX 代表 theta)
        df_group = read_obs_out(obs_filepath, target_variable=TARGET_VAR)

        if df_group is None or df_group.empty:
            print(f"警告: 未能从 {obs_filepath} 读取或提取有效数据。跳过该分组。")
            continue

        # --- 筛选出时间为整数的数据行 ---
        df_group[TIME_COLUMN_NAME] = pd.to_numeric(df_group[TIME_COLUMN_NAME], errors='coerce')
        df_group = df_group.dropna(subset=[TIME_COLUMN_NAME])
        df_group_filtered = df_group[df_group[TIME_COLUMN_NAME] % 1 == 0].copy()
        # print(f"    筛选出 {len(df_group_filtered)} 行整数时间数据。") # 细化提示

        if df_group_filtered.empty:
            print(f"警告: 文件 {obs_filepath} 在筛选整数时间后没有剩余数据。跳过该分组。")
            continue
        # --- 筛选结束 ---

        # 移除重复的时间点，针对整数时间
        df_group_processed = df_group_filtered.drop_duplicates(subset=[TIME_COLUMN_NAME]).set_index(TIME_COLUMN_NAME)

        # 重命名列以包含分组信息 PointX -> PointX_GroupY
        cols_to_rename = {col: f"{col}_{group}" for col in df_group_processed.columns if col.startswith('Point')}
        df_group_final = df_group_processed.rename(columns=cols_to_rename)

        # Note: We don't filter by mapping_df here yet.
        # We merge all points read from the file that match the PointX pattern.
        # The mapping_df will be used later in get_data_at_time to link PointX_GroupY to physical coords.

        if merged_df is None:
            merged_df = df_group_final
        else:
            # 合并DataFrame，时间是索引
            merged_df = merged_df.join(df_group_final, how='outer', lsuffix='_left', rsuffix='_right')
            # 处理join可能带来的重复列（不应发生，但以防万一）
            merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]


    if merged_df is not None:
        merged_df = merged_df.sort_index()
        # 筛选掉在mapping_df中完全找不到对应物理坐标的列，避免内存浪费和后续警告
        # Get all possible PointX_GroupY identifiers from mapping_df for this profile and its groups
        valid_point_ids_in_mapping = []
        for group in obs_groups:
             group_mapping = mapping_df[(mapping_df['Profile'] == profile) & (mapping_df['Group'] == group)]
             valid_point_ids_in_mapping.extend([f'Point{pid}_{group}' for pid in group_mapping['ObsPointID']])

        # Filter merged_df columns to keep only Time index and columns found in valid_point_ids_in_mapping
        cols_to_keep = [col for col in merged_df.columns if col in valid_point_ids_in_mapping]
        # Ensure we don't end up with an empty DataFrame if no points match
        if not cols_to_keep:
             print(f"警告: 对于 {land_use}, 剖面 {profile}, 合并后的数据中没有列与映射文件中的任何点匹配。生成空数据。")
             return pd.DataFrame() # Return empty if no points match mapping

        merged_df = merged_df[cols_to_keep]


        print(f"成功合并 {land_use}, 剖面 {profile} 的数据 (仅整数时间)。总时间步: {len(merged_df)}, 匹配映射点: {merged_df.shape[1]} 个 {TARGET_VAR} 数据序列。")

    return merged_df

# --- 函数：加载并合并所有数据 ---
# (此函数与之前基本相同，调用 merge_profile_data)
def load_all_simulation_data(project_dir, profile_list, group_list, land_use_list, mapping_df):
    """
    遍历所有土地利用和剖面，加载并合并Obs.out数据。
    返回一个嵌套字典: data[land_use][profile] = merged_dataframe。
    """
    all_data = {}
    print("\n--- 开始加载和合并模拟数据 ---")
    for land_use in land_use_list:
        all_data[land_use] = {}
        print(f"\n正在处理土地利用: '{land_use}'")
        for profile in profile_list:
            # print(f"  正在处理剖面: '{profile}'") # 提示级别可以更低，由 merge_profile_data 内部提示读取文件
            merged_df = merge_profile_data(project_dir, profile, land_use, group_list, mapping_df)
            if merged_df is not None and not merged_df.empty:
                all_data[land_use][profile] = merged_df
            else:
                 print(f"警告: 未为 {land_use}, 剖面 {profile} 生成有效合并数据。")

    print("\n数据加载及合并完成。")
    return all_data

# --- 函数：提取特定时间点的所有数据点 ---
def get_data_at_time(all_data, mapping_df, land_use, time_point):
    """
    提取指定土地利用在特定时间点的所有观测点数据 (含水量及物理坐标)。
    返回一个DataFrame，包含 PhysicalX, PhysicalY, PhysicalZ, WaterContent, Profile, Group, ObsPointID。
    """
    all_points_data = []

    if land_use not in all_data or not all_data[land_use]:
         print(f"错误: 未找到 '{land_use}' 土地利用或其下无数据。")
         return pd.DataFrame()

    # 遍历该土地利用下的所有剖面数据
    for profile, merged_df in all_data[land_use].items():
        if merged_df is None or merged_df.empty:
            continue

        # 找到该剖面数据中最接近指定时间点的数据行
        try:
            # get_loc with method='nearest' works on the index
            closest_time_index_pos = merged_df.index.get_loc(time_point, method='nearest')
            time_slice_df = merged_df.iloc[[closest_time_index_pos]]
            actual_time = merged_df.index[closest_time_index_pos] # 实际找到的时间值

            # # 可选：检查实际时间与输入时间的差异
            # if abs(actual_time - time_point) > 1e-6:
            #     print(f"  对于剖面 {profile}, 指定时间 {time_point:.2f} 非精确整数或无数据，使用最接近时间 {actual_time:.2f}")


        except KeyError:
             # print(f"警告: 对于剖面 {profile}, 指定时间 {time_point:.2f} 超出数据范围或无法找到最近时间。跳过。") # 避免过多警告
             continue # 跳过没有该时间数据的剖面
        except Exception as e:
             print(f"提取剖面 {profile} 时间切片时发生错误: {e}。跳过。")
             continue


        # 提取该时间点该剖面所有观测点的数据
        # water_content_row 的列名是 PointX_GroupY
        water_content_row = time_slice_df.iloc[0].dropna() # 移除NaN值（不应有，但安全起见）

        # 迭代提取到的含水量数据，查找其物理坐标并收集
        for col_name, wc_value in water_content_row.items():
            # 解析列名获取原始点ID和分组，例如 Point1_A -> (1, 'A')
            match = re.match(r'Point(\d+)_([ABCD])', col_name)
            if match:
                obs_point_id = int(match.group(1))
                group = match.group(2)

                # 在映射表中查找对应的物理坐标 (使用 Profile, Group, ObsPointID)
                point_info = mapping_df[
                    (mapping_df['Profile'] == profile) &
                    (mapping_df['Group'] == group) &
                    (mapping_df['ObsPointID'] == obs_point_id)
                ]

                if not point_info.empty:
                     # 确保只有一个匹配项
                    if len(point_info) > 1:
                         print(f"警告: 映射表中存在多个匹配项 for Point {obs_point_id}, Group {group}, Profile {profile}。使用第一个。")

                    point_data = {
                        'Profile': profile,
                        'Group': group,
                        'ObsPointID': obs_point_id,
                        'PhysicalColumn': point_info['PhysicalColumn'].iloc[0],
                        'PhysicalDepth': point_info['PhysicalDepth'].iloc[0],
                        'PhysicalX': point_info['PhysicalX'].iloc[0],
                        'PhysicalY': point_info['PhysicalY'].iloc[0],
                        'PhysicalZ': point_info['PhysicalZ'].iloc[0],
                        'WaterContent': wc_value,
                        'ActualTime': actual_time # 记录实际找到的时间
                    }
                    all_points_data.append(point_data)
                else:
                    # 这个点在合并的数据中存在，但在映射表中找不到，可能是映射文件不完整
                    # print(f"警告: 未在映射表中找到 {col_name} (Profile: {profile}, Group: {group}, ObsPointID: {obs_point_id}) 的物理坐标。") # 避免过多警告
                    pass # 不打印太多警告，只收集找到的有效点
            else:
                # print(f"警告: 无法解析列名 '{col_name}' 以提取点信息。") # 避免过多警告
                pass # 不打印太多警告

    if not all_points_data:
        # print(f"错误: 在土地利用 '{land_use}', 时间点 {time_point:.2f} 没有找到可以用于可视化的观测点数据。请检查映射文件和数据。") # 错误提示已在调用处处理
        return pd.DataFrame() # 返回空DataFrame表示无数据

    return pd.DataFrame(all_points_data)


# --- 函数：可视化单个剖面在特定时间的含水量分布 (二维) ---
def visualize_profile_water_content_2d(all_points_data_at_time_df, land_use, profile, actual_time):
    """
    可视化指定土地利用、剖面在特定时间点的含水量分布 (二维)。
    使用 PhysicalColumn 和 PhysicalDepth 进行绘图。
    """
    # 从所有点数据中筛选出属于当前剖面的点
    profile_points_df = all_points_data_at_time_df[all_points_data_at_time_df['Profile'] == profile]

    if profile_points_df.empty:
        print(f"错误: 在土地利用 '{land_use}', 剖面 '{profile}', 时间点 {actual_time:.2f} 没有找到数据用于二维剖面可视化。")
        return

    x = profile_points_df['PhysicalColumn']
    y = profile_points_df['PhysicalDepth']
    z = profile_points_df['WaterContent']

    # 创建散点图
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=z, cmap='viridis', s=100, edgecolors='k', alpha=0.7)
    plt.colorbar(scatter, label=f'{TARGET_VAR} (Volumetric Water Content)') # 根据提取的变量修改标签
    plt.xlabel('Horizontal Position (Column Index)') # 或根据实际单位修改
    plt.ylabel('Depth (m)') # 根据实际单位修改
    plt.title(f'2D {TARGET_VAR} Profile: {land_use} - Profile {profile} at Time {actual_time:.2f}')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 反转Y轴，使深度向下增加
    plt.gca().invert_yaxis()

    # 设置X轴刻度
    if x.dtype in [np.int64, np.int32, float] and len(np.unique(x)) < 20:
         plt.xticks(np.unique(x))

    plt.tight_layout()
    plt.show()


# --- 函数：可视化整个区域在特定时间的含水量分布 (三维) ---
def visualize_water_content_3d(all_points_data_at_time_df, land_use, actual_time):
    """
    可视化指定土地利用在特定时间点的所有观测点含水量分布 (三维)。
    使用 PhysicalX, PhysicalY, PhysicalZ 进行绘图。
    """
    if all_points_data_at_time_df.empty:
        print(f"错误: 在土地利用 '{land_use}', 时间点 {actual_time:.2f} 没有找到数据用于三维可视化。")
        return

    x = all_points_data_at_time_df['PhysicalX']
    y = all_points_data_at_time_df['PhysicalY']
    z = all_points_data_at_time_df['PhysicalZ']
    c = all_points_data_at_time_df['WaterContent'] # 用颜色表示含水量

    # 创建三维散点图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', marker='o', s=50) # s控制点的大小

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(f'{TARGET_VAR} (Volumetric Water Content)') # 根据提取的变量修改标签

    # 设置轴标签
    ax.set_xlabel('X Position (m)') # 根据您的PhysicalX/Y/Z单位修改
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')

    ax.set_title(f'3D {TARGET_VAR} Distribution: {land_use} at Time {actual_time:.2f}')

    # 可选：反转Z轴，使深度向下增加 (如果 PhysicalZ 向上为正，且您希望向下为深度)
    # ax.invert_zaxis()

    # 可选：调整视角
    # ax.view_init(elev=20., azim=-35)

    plt.tight_layout()
    plt.show()

# --- 函数：可视化映射文件中的点 (三维验证) ---
def verify_mapping_file_3d(mapping_df):
    """
    可视化观测点映射文件中定义的所有点在三维空间中的位置。
    """
    if mapping_df is None or mapping_df.empty:
        print("映射文件数据为空，无法进行验证可视化。")
        return

    x = mapping_df['PhysicalX']
    y = mapping_df['PhysicalY']
    z = mapping_df['PhysicalZ']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 可以按剖面或分组给点上色以便区分
    # 例如按剖面上色
    unique_profiles = mapping_df['Profile'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_profiles)))
    profile_color_map = dict(zip(unique_profiles, colors))

    for profile in unique_profiles:
        profile_df = mapping_df[mapping_df['Profile'] == profile]
        ax.scatter(profile_df['PhysicalX'], profile_df['PhysicalY'], profile_df['PhysicalZ'],
                   color=profile_color_map[profile], label=f'Profile {profile}', s=50)

    # 如果点数不多，可以考虑添加文本标签 (可能会非常拥挤)
    # for index, row in mapping_df.iterrows():
    #    ax.text(row['PhysicalX'], row['PhysicalY'], row['PhysicalZ'],
    #            f"{row['Profile']}-{row['Group']}-{row['ObsPointID']}", size=8, zorder=1, color='k')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Visualization of Observation Points from Mapping File')

    # ax.invert_zaxis() # 如果需要反转Z轴

    ax.legend() # 显示剖面图例
    plt.tight_layout()
    plt.show()


# --- 主执行流程 ---
if __name__ == "__main__":
    # 1. 加载观测点映射文件
    obs_mapping_df = load_obs_point_mapping(OBS_POINT_MAPPING_FILE)
    if obs_mapping_df is None:
        print("无法继续，观测点映射文件加载失败。")
    else:
        # 提供验证映射文件的选项
        print("\n请选择操作:")
        print("1. 验证观测点映射文件 (3D可视化)")
        print("2. 处理模拟数据并可视化")
        print("quit - 退出")
        initial_choice = input("请输入选项 (1, 2, 或 quit): ").strip().lower()

        if initial_choice == '1':
            verify_mapping_file_3d(obs_mapping_df)
        elif initial_choice == '2':
            # 2. 加载并合并所有模拟数据
            # 这步可能需要一些时间
            all_merged_data = load_all_simulation_data(
                PROJECT_ROOT_DIR,
                PROFILE_LOCATIONS,
                OBS_GROUPS,
                LAND_USE_TYPES,
                obs_mapping_df # Pass mapping_df here to filter merged columns
            )

            # 3. 允许用户交互式可视化
            print("\n--- 可视化模式 ---")
            while True:
                print("\n请选择可视化类型:")
                print("1. 2D 剖面视图")
                print("2. 3D 整体视图")
                print("quit - 退出")

                viz_type = input("请输入选项 (1, 2, 或 quit): ").strip().lower()

                if viz_type == 'quit':
                    break
                elif viz_type not in ['1', '2']:
                    print("输入无效，请重新输入。")
                    continue

                print("\n可用土地利用类型:", ", ".join(LAND_USE_TYPES))
                selected_land_use = input("请输入要查看的土地利用类型: ").strip()

                if selected_land_use not in all_merged_data or not all_merged_data[selected_land_use]:
                    print("输入无效，请从可用列表中选择，或该土地利用下无数据。")
                    continue

                # 提示用户输入时间点 (对 2D 和 3D 都需要)
                # 找到该土地利用下所有数据的时间范围，作为提示
                min_time = float('inf')
                max_time = float('-inf')
                # Iterate through all profiles for this land use to find min/max time
                for profile_data in all_merged_data[selected_land_use].values():
                     if profile_data is not None and not profile_data.empty:
                         min_time = min(min_time, profile_data.index.min())
                         max_time = max(max_time, profile_data.index.max())

                if min_time == float('inf'): # Still infinity means no data found for this land use
                     print(f"错误: 对于土地利用 '{selected_land_use}', 未加载到任何有效合并数据。")
                     continue # Go back to start of visualization loop

                print(f"该土地利用下数据的时间范围 (仅整数天) 从 {min_time:.2f} 到 {max_time:.2f}")
                time_input = input(f"请输入要查看的整数时间点 (模拟时间数值，例如 {int((min_time+max_time)/2) if not pd.isna((min_time+max_time)/2) else '某个整数'}): ").strip() # Handle potential NaN if min/max are inf

                try:
                     selected_time = float(time_input)
                except ValueError:
                     print("时间输入无效，请输入一个数值。")
                     continue

                # 提取该时间和土地利用下的所有点数据
                print(f"\n正在提取土地利用 '{selected_land_use}' 在时间 {selected_time:.2f} 附近的所有观测点数据...")
                all_points_data_at_selected_time = get_data_at_time(
                     all_merged_data,
                     obs_mapping_df,
                     selected_land_use,
                     selected_time # 传入用户输入的数值
                 )

                if all_points_data_at_selected_time.empty:
                     print(f"无法进行可视化，因为在土地利用 '{selected_land_use}', 时间点 {selected_time:.2f} 没有找到有效的观测点数据。")
                     continue # Go back to start of visualization loop

                # 获取实际找到的时间点
                actual_time_used = all_points_data_at_selected_time['ActualTime'].iloc[0]

                if viz_type == '1': # 2D Profile View
                    available_profiles = list(all_merged_data[selected_land_use].keys())
                    if not available_profiles: # Should not happen if all_points_data_at_selected_time is not empty
                         print(f"对于土地利用 '{selected_land_use}', 未找到任何可用剖面。")
                         continue

                    print(f"可用剖面位置 ({selected_land_use}):", ", ".join(available_profiles))
                    selected_profile = input("请输入要查看的剖面位置 (例如 '15'): ").strip()

                    if selected_profile not in available_profiles:
                        print("输入无效，请从可用列表中选择。")
                        continue

                    # 进行 2D 可视化
                    visualize_profile_water_content_2d(
                        all_points_data_at_selected_time, # 传入已提取的所有点数据
                        selected_land_use,
                        selected_profile,
                        actual_time_used # 传入实际使用的时间
                    )

                elif viz_type == '2': # 3D Volume View
                     # 进行 3D 可视化
                     visualize_water_content_3d(
                         all_points_data_at_selected_time, # 传入已提取的所有点数据
                         selected_land_use,
                         actual_time_used # 传入实际使用的时间
                     )
        elif initial_choice != 'quit':
             print("输入无效，请重新运行并选择 1, 2, 或 quit。")


    print("程序结束。")