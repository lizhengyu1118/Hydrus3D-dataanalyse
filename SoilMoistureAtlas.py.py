import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from mpl_toolkits.mplot3d import Axes3D
# import joblib # Used for saving/loading mapping if needed, or keep for future use # No, using for parallelism now
from joblib import Parallel, delayed # Import Parallel and delayed
from datetime import datetime, timedelta
import xarray as xr # 导入 Xarray
import netCDF4 # Xarray backend for NetCDF
import traceback # For more detailed error reporting if needed

# --- 配置参数 ---
PROJECT_ROOT_DIR = './Hydrus3D_Simulations' # 您的项目根目录路径
OBS_POINT_MAPPING_FILE = 'obs_point_mapping.csv' # 观测点坐标映射文件路径
PROFILE_LOCATIONS = ['05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55'] # 您的剖面位置列表
OBS_GROUPS = ['A', 'B', 'C', 'D'] # 您的观测点分组列表
LAND_USE_TYPES = ['exoticgrass', 'naturalgrass', 'exoticshrub', 'arablecrop'] # 您的土地利用类型列表
OBS_OUT_FILENAME = 'Obs.out' # Hydrus观测点输出文件名
TIME_COLUMN_NAME = 'Time' # Obs.out 文件中的时间列名 (模拟天数)
TARGET_VAR = 'theta' # 我们要提取的变量名称

# --- 中间文件配置 ---
INTERMEDIATE_DATA_DIR = './intermediate_xarray_data' # 存放 Xarray 中间文件的目录
# 中间文件命名规则: 每个土地利用一个文件
INTERMEDIATE_FILE_TEMPLATE = '{land_use}.nc' # 使用 NetCDF 格式

# --- 日期配置 ---
# 模拟开始的第一天
SIM_START_DATE = datetime(2004, 7, 8)

# --- Joblib 配置 ---
N_JOBS = -1 # 使用所有可用的 CPU 核心 (-1). 设置为 1 可禁用并行处理进行调试.

# --- 函数：模拟天数转实际日期 ---
def sim_day_to_date(sim_day):
    """
    将模拟天数转换为实际日期。
    """
    if pd.isna(sim_day): return pd.NaT # Handle potential NaN
    # Handle potential non-integer days if necessary, though we filter for integers later
    # Using floor or round might be alternatives depending on desired behavior for non-int days
    try:
        return SIM_START_DATE + timedelta(days=int(np.floor(sim_day)))
    except (ValueError, TypeError):
        return pd.NaT

# --- 函数：实际日期转模拟天数 ---
def date_to_sim_day(target_date_str):
    """
    将 YYYY-MM-DD 格式的日期字符串转换为模拟天数（整数天）。
    """
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        time_delta = target_date - SIM_START_DATE
        return time_delta.days
    except ValueError:
        print(f"错误: 日期格式应为 YYYY-MM-DD，例如 2004-07-08。无法转换 '{target_date_str}'。")
        return None
    except TypeError:
         print("错误: 输入的日期格式不正确。")
         return None

# --- 函数：加载观测点映射文件 ---
# (No changes needed here)
def load_obs_point_mapping(mapping_filepath):
    """
    加载观测点ID到物理坐标的映射文件。
    返回一个DataFrame。
    """
    try:
        mapping_df = pd.read_csv(mapping_filepath)
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
# (No changes needed here)
def read_obs_out(filepath, target_variable=TARGET_VAR):
    """
    读取Hydrus的Obs.out文件，跳过注释行和Node行，提取指定变量列。
    并按顺序将变量列重命名为 Point1, Point2, ...
    """
    # This function remains the same as it's called by merge_profile_data
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        node_line_index = -1
        for i, line in enumerate(lines):
            # Improved check for potential empty lines after comments but before node line
            stripped_line = line.strip()
            if not stripped_line.startswith('#') and stripped_line:
                # Check if it looks like the node line (usually starts with 'Node' or similar)
                # This is heuristic; adjust if needed based on actual file format variations
                if stripped_line.lower().startswith('node') or re.match(r'^\s*\d+', stripped_line):
                    node_line_index = i
                    break
                else:
                     # If it's not a comment and not the node line, maybe it's an unexpected header?
                     # Or maybe the file starts directly with the header. Let's assume header if no node line found soon.
                     # This part might need refinement based on exact Obs.out variations.
                     # For now, stick to the original logic assuming a clear node line exists before the header.
                     pass


        if node_line_index == -1:
             # Try finding header directly if no node line detected
             for i, line in enumerate(lines):
                stripped_line = line.strip()
                if not stripped_line.startswith('#') and stripped_line:
                    # Assume the first non-comment, non-empty line is the header if no 'Node' line found
                    header_line_index = i
                    print(f"警告: 文件 {filepath} 未明确找到 'Node' 行，尝试从第 {header_line_index+1} 行作为表头读取。")
                    break
             else: # No non-comment, non-empty line found at all
                 print(f"警告: 文件 {filepath} 似乎为空或只包含注释。")
                 return None

        else: # Node line was found, search for header after it
            header_line_index = -1
            for i in range(node_line_index + 1, len(lines)):
                if not lines[i].strip().startswith('#') and lines[i].strip():
                    header_line_index = i
                    break

        if header_line_index == -1:
            print(f"警告: 文件 {filepath} 在 Node 行后未找到表头行。")
            return None

        # Robust reading with error handling for bad lines
        df = pd.read_csv(filepath, sep='\s+', skiprows=header_line_index, header=0, comment='#', on_bad_lines='warn')
        df = df.dropna(axis=1, how='all') # Drop columns that are entirely NaN

        # Check if TIME_COLUMN_NAME exists after potential bad lines handling
        if TIME_COLUMN_NAME not in df.columns:
            # Sometimes the header might be misaligned, try finding 'Time' like column
            time_like_cols = [col for col in df.columns if isinstance(col, str) and 'time' in col.lower()]
            if time_like_cols:
                actual_time_col = time_like_cols[0]
                print(f"警告: 文件 {filepath} 未找到名为 '{TIME_COLUMN_NAME}' 的列，但找到 '{actual_time_col}'。将使用此列作为时间。")
                df = df.rename(columns={actual_time_col: TIME_COLUMN_NAME})
            else:
                print(f"错误: 文件 {filepath} 读取后未找到时间列 ('{TIME_COLUMN_NAME}' 或类似名称)。列: {df.columns}")
                return None

        if df.shape[1] < 2:
             print(f"警告: 文件 {filepath} 读取后少于两列 (需要时间和至少一个数据列)。")
             return None

        # Ensure Time column is numeric, coercing errors
        df[TIME_COLUMN_NAME] = pd.to_numeric(df[TIME_COLUMN_NAME], errors='coerce')
        df = df.dropna(subset=[TIME_COLUMN_NAME]) # Drop rows where time could not be parsed

        time_col_data = df[TIME_COLUMN_NAME]

        # Find target variable columns - more robustly handle potential variations like 'theta(1)'
        target_cols_data = df.filter(regex=f'^{re.escape(target_variable)}(\(\d+\))?$', axis=1)

        if target_cols_data.empty:
            print(f"警告: 文件 {filepath} 未找到名为 '{target_variable}' 或类似模式的列。 可用列: {df.columns.tolist()}")
            return None

        # Rename columns sequentially
        new_point_col_names = [f'Point{i+1}' for i in range(target_cols_data.shape[1])]
        target_cols_data.columns = new_point_col_names

        processed_df = pd.concat([time_col_data, target_cols_data], axis=1)

        return processed_df

    except FileNotFoundError:
        # Suppress this print when called in parallel, let the main loop handle overall progress
        # print(f"文件未找到: {filepath}")
        return None
    except pd.errors.EmptyDataError:
         # print(f"警告: 文件 {filepath} 为空或无法解析。")
        return None
    except Exception as e:
        print(f"读取文件 {filepath} 时发生意外错误: {e}")
        # traceback.print_exc() # Uncomment for detailed traceback during debugging
        return None


# --- 函数：合并一个剖面的所有分组数据 (返回 Pandas DataFrame) ---
# (No fundamental changes needed for parallelism, but added prints for clarity)
def merge_profile_data(base_dir, profile, land_use, obs_groups, mapping_df):
    """
    合并指定土地利用类型和剖面位置下的所有分组(A, B, C, D)的Obs.out数据。
    返回一个包含所有观测点时间序列的DataFrame (Pandas)。
    同时筛选出时间为整数天的数据。
    (Designed to be called potentially in parallel by joblib)
    """
    # print(f"  [Worker] 开始处理: {land_use}, 剖面 {profile}") # Informative print for parallel execution
    merged_df = None
    profile_point_count = 0
    profile_timestep_count = 0

    for group in obs_groups:
        # Use a more specific regex to avoid accidental matches if folder names are complex
        # Match digits (6-8) for date, hyphen, profile, hyphen, group, hyphen, land_use, end-of-string
        pattern = re.compile(r'^\d{6,8}-' + re.escape(profile) + '-' + re.escape(group) + '-' + re.escape(land_use) + r'$')
        matching_dirs = []
        try:
            # Check if base_dir exists before listing
            if not os.path.isdir(base_dir):
                print(f"错误: 项目根目录未找到或不是目录: {base_dir}")
                return None # Cannot proceed if root directory is invalid

            for d in os.listdir(base_dir):
                dir_path = os.path.join(base_dir, d)
                if os.path.isdir(dir_path) and pattern.match(d):
                    matching_dirs.append(d)
        except FileNotFoundError:
             print(f"错误: 无法访问项目根目录: {base_dir}")
             return None # Stop processing this profile if root dir is inaccessible
        except Exception as e:
             print(f"列出目录 {base_dir} 时出错: {e}")
             return None # Stop processing this profile on other listing errors

        if not matching_dirs:
            # print(f"    信息: 未找到匹配目录 {pattern.pattern} 在 {base_dir}")
            continue # Skip this group if no matching directory found

        # Handle case where multiple directories might match (e.g., different run dates)
        # Here, we take the first one found. Consider sorting or specific selection logic if needed.
        sim_folder = matching_dirs[0]
        if len(matching_dirs) > 1:
            print(f"    警告: 找到多个匹配目录 {pattern.pattern}，使用第一个: {sim_folder}")

        obs_filepath = os.path.join(base_dir, sim_folder, OBS_OUT_FILENAME)

        if not os.path.exists(obs_filepath):
             # print(f"    信息: 文件 {obs_filepath} 不存在，跳过分组 {group}")
             continue

        # Call the robust read_obs_out function
        df_group = read_obs_out(obs_filepath, target_variable=TARGET_VAR)

        if df_group is None or df_group.empty:
            # Warning already printed in read_obs_out or here if None/empty
            # print(f"    警告: 未能从 {obs_filepath} 读取或提取有效数据。跳过分组 {group}。")
            continue

        # --- 筛选出时间为整数的数据行 ---
        # Time column already converted to numeric with coercion in read_obs_out
        # df_group[TIME_COLUMN_NAME] = pd.to_numeric(df_group[TIME_COLUMN_NAME], errors='coerce') # Done in read_obs_out
        # df_group = df_group.dropna(subset=[TIME_COLUMN_NAME]) # Done in read_obs_out

        # Check for floating point precision issues when checking for integers
        # Use a small tolerance instead of direct equality comparison
        tolerance = 1e-9
        is_integer_day = np.abs(df_group[TIME_COLUMN_NAME] - np.round(df_group[TIME_COLUMN_NAME])) < tolerance
        df_group_filtered = df_group[is_integer_day].copy()

        if df_group_filtered.empty:
            # print(f"    信息: 文件 {obs_filepath} 中未找到整数时间步的数据。")
            continue
        # --- 筛选结束 ---

        # Ensure time column is integer after filtering (using round to avoid precision issues)
        df_group_filtered[TIME_COLUMN_NAME] = np.round(df_group_filtered[TIME_COLUMN_NAME]).astype(int)

        # Drop duplicate time steps (e.g., if multiple outputs for the exact same integer day exist)
        # Keep the first occurrence
        df_group_processed = df_group_filtered.drop_duplicates(subset=[TIME_COLUMN_NAME]).set_index(TIME_COLUMN_NAME)

        # Rename columns to include group identifier
        cols_to_rename = {col: f"{col}_{group}" for col in df_group_processed.columns if col.startswith('Point')}
        df_group_final = df_group_processed.rename(columns=cols_to_rename)

        if merged_df is None:
            merged_df = df_group_final
        else:
            # Use outer join to keep all time steps from both dataframes
            merged_df = merged_df.join(df_group_final, how='outer', lsuffix='_left_dup', rsuffix='_right_dup')
            # Remove potential duplicate columns created by join if column names were identical before adding suffix (shouldn't happen with our renaming)
            merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
            # Remove the temporary duplicate suffixes if they somehow occurred
            merged_df.columns = merged_df.columns.str.replace('_left_dup$', '', regex=True).str.replace('_right_dup$', '', regex=True)


    if merged_df is not None and not merged_df.empty:
        merged_df = merged_df.sort_index() # Sort by time index

        # --- Filter columns based on mapping file ---
        # Find all valid point IDs for *this specific profile* from the mapping file
        valid_point_ids_in_mapping = []
        profile_mapping = mapping_df[mapping_df['Profile'] == profile]
        if not profile_mapping.empty:
             for group in obs_groups:
                 group_mapping = profile_mapping[profile_mapping['Group'] == group]
                 valid_point_ids_in_mapping.extend([f'Point{pid}_{group}' for pid in group_mapping['ObsPointID']])

        # Find which columns in the merged data actually exist in the valid list
        cols_to_keep = [col for col in merged_df.columns if col in valid_point_ids_in_mapping]

        if not cols_to_keep:
             # print(f"    警告: {land_use}, 剖面 {profile} - 合并数据中的列均未在映射文件中找到对应点。")
             return pd.DataFrame() # Return empty DataFrame if no points match mapping

        # Keep only the valid columns
        merged_df = merged_df[cols_to_keep]
        profile_point_count = merged_df.shape[1]
        profile_timestep_count = len(merged_df)

        # Suppress print here, let the main parallel loop summarize
        # print(f"成功合并 {land_use}, 剖面 {profile} 的数据 (仅整数时间)。总时间步: {profile_timestep_count}, 匹配映射点: {profile_point_count} 个 {TARGET_VAR} 数据序列。")
        return merged_df
    else:
        # print(f"  [Worker] 完成处理 (无有效数据): {land_use}, 剖面 {profile}")
        return pd.DataFrame() # Return empty DataFrame if no data was merged at all


# --- 函数：将合并后的 Pandas DataFrames 转换为 Xarray Dataset ---
# (No changes needed here)
def convert_to_xarray_dataset(profile_dataframes, mapping_df, land_use):
    """
    将一个土地利用下的所有剖面 Pandas DataFrames (时间索引, PointX_GroupY 列)
    转换为一个 Xarray Dataset.
    Added land_use parameter for better logging.
    """
    all_points_series = []
    point_metadata = []

    # Collect all time series and their metadata across all profiles for this land_use
    for profile, merged_df in profile_dataframes.items():
        if merged_df is None or merged_df.empty: continue

        # Iterate through columns (PointX_GroupY)
        for col_name in merged_df.columns:
             match = re.match(r'Point(\d+)_([ABCD])', col_name)
             if match:
                 obs_point_id = int(match.group(1))
                 group = match.group(2)

                 # Find point metadata from mapping_df
                 # Ensure profile is matched correctly (should be, as it came from profile_dataframes key)
                 point_info = mapping_df[
                     (mapping_df['Profile'] == profile) &
                     (mapping_df['Group'] == group) &
                     (mapping_df['ObsPointID'] == obs_point_id)
                 ]

                 if not point_info.empty:
                     # Store the time series data (Pandas Series)
                     all_points_series.append(merged_df[col_name])
                     # Store the metadata for this point (as dict)
                     point_metadata.append(point_info.iloc[0].to_dict())
                 # else: warning would have been printed during merge_profile_data column filtering

    if not all_points_series:
        print(f"警告: 土地利用 '{land_use}' 没有找到可以转换为 Xarray 的有效观测点时间序列。")
        return None

    # Concatenate all time series into a single Pandas DataFrame
    # Using pd.concat with axis=1 and join='outer' ensures all time steps are included
    # This step can be memory-intensive if there are many points and time steps
    try:
        print(f"  正在合并 {len(all_points_series)} 个观测点时间序列...")
        all_series_df = pd.concat(all_points_series, axis=1, join='outer')
        print(f"  合并后 DataFrame 形状: {all_series_df.shape}")
    except MemoryError:
        print(f"错误: 合并土地利用 '{land_use}' 的所有时间序列时内存不足。可能需要更多 RAM 或处理更少的数据。")
        return None
    except Exception as e:
        print(f"合并时间序列时出错 for land_use '{land_use}': {e}")
        # traceback.print_exc()
        return None


    # Create a unique identifier for each point for the Xarray dimension
    # Using a simple integer index for the 'point' dimension is easiest
    point_index = pd.RangeIndex(len(point_metadata))

    # Create coordinates for the 'point' dimension from the collected metadata
    point_coords_df = pd.DataFrame(point_metadata, index=point_index)

    # Ensure time index is sorted and has a name
    # The index should already be sorted from merge_profile_data and unique after concat
    time_index = all_series_df.index
    if not time_index.is_monotonic_increasing:
        time_index = time_index.sort_values()
        all_series_df = all_series_df.reindex(index=time_index) # Reindex df if time was not sorted
    time_index.name = 'sim_day' # Name the time index

    # Create actual date coordinate
    print(f"  正在转换 {len(time_index)} 个模拟天数为日期...")
    date_coords = [sim_day_to_date(day) for day in time_index]

    # Convert the combined time series DataFrame to a NumPy array for Xarray DataArray
    # Ensure alignment with the (potentially sorted) time_index
    print(f"  正在将 Pandas 数据转换为 NumPy 数组...")
    water_content_data_array = all_series_df.values # Values should now align with time_index

    # Create the Xarray DataArray
    print(f"  正在创建 Xarray DataArray '{TARGET_VAR}'...")
    wc_data_array = xr.DataArray(
        water_content_data_array,
        coords=[('sim_day', time_index), ('point', point_index)], # Define coordinates along dimensions
        dims=['sim_day', 'point'],                                # Name the dimensions
        name=TARGET_VAR,                                          # Variable name in Dataset
        attrs={'long_name': f'Volumetric Water Content ({TARGET_VAR})', 'units': '-'} # Add attributes
    )

    # Create the Xarray Dataset
    print(f"  正在创建 Xarray Dataset...")
    dataset = xr.Dataset({TARGET_VAR: wc_data_array})

    # Add point metadata as coordinates associated with the 'point' dimension
    print(f"  正在添加观测点坐标信息...")
    for col in point_coords_df.columns:
         # Ensure coordinates have the correct dimension ('point')
         dataset[col] = ('point', point_coords_df[col].values)

    # Add actual date as coordinate associated with the 'sim_day' dimension
    print(f"  正在添加日期坐标...")
    dataset['date'] = ('sim_day', date_coords)

    # Add dataset attributes
    dataset.attrs['description'] = f"Hydrus-3D {TARGET_VAR} data for {land_use} land use (processed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    dataset.attrs['hydrus_project_dir'] = PROJECT_ROOT_DIR
    dataset.attrs['mapping_file'] = OBS_POINT_MAPPING_FILE
    dataset.attrs['target_variable'] = TARGET_VAR
    dataset.attrs['simulation_start_date'] = SIM_START_DATE.strftime('%Y-%m-%d')


    print(f"  为 '{land_use}' 成功转换为 Xarray Dataset。维度: {dataset.dims}, 变量: {list(dataset.variables.keys())}")
    return dataset


# --- 函数：加载并合并所有数据 (优化版：使用 Xarray 中间文件 + Joblib 并行处理) ---
def load_all_simulation_data_optimized_xarray(project_dir, profile_list, group_list, land_use_list, mapping_df, intermediate_dir=INTERMEDIATE_DATA_DIR, force_reprocess=False):
    """
    遍历所有土地利用，处理或加载 Xarray 中间文件。
    如果需要重新处理，则使用 joblib 并行处理各个剖面。
    返回一个嵌套字典: data[land_use] = Xarray Dataset。
    """
    all_data = {}
    os.makedirs(intermediate_dir, exist_ok=True) # Ensure intermediate directory exists

    print("\n--- 开始加载或处理模拟数据 (使用 Xarray 中间文件 / Joblib 并行) ---")
    if force_reprocess:
        print(f"将强制重新处理所有数据 (使用 {N_JOBS if N_JOBS != -1 else '所有'} CPU 核心)，忽略已存在的中间文件。")
    else:
         print(f"将尝试从中间文件加载，如果失败或文件不存在则处理数据 (使用 {N_JOBS if N_JOBS != -1 else '所有'} CPU 核心)。")


    for land_use in land_use_list:
        intermediate_filepath = os.path.join(intermediate_dir, INTERMEDIATE_FILE_TEMPLATE.format(land_use=land_use))
        dataset = None # Reset for each land use

        if os.path.exists(intermediate_filepath) and not force_reprocess:
            # --- Attempt to load from intermediate file (Xarray Dataset) ---
            try:
                print(f"\n发现中间文件: {intermediate_filepath}, 正在加载...")
                dataset = xr.open_dataset(intermediate_filepath, engine='netcdf4') # Specify engine for clarity

                # Basic validation of loaded dataset
                if dataset is not None and TARGET_VAR in dataset.variables and 'sim_day' in dataset.dims and 'point' in dataset.dims:
                    all_data[land_use] = dataset
                    print(f"成功加载 {land_use} 数据从中间文件。形状: {dataset[TARGET_VAR].shape}, 维度: {list(dataset.dims)}")
                else:
                    print(f"警告: 从中间文件 {intermediate_filepath} 加载的数据无效或不完整 (缺少变量 '{TARGET_VAR}' 或维度 'sim_day'/'point')。将尝试重新处理。")
                    if dataset: dataset.close() # Close the invalid file
                    dataset = None # Mark for reprocessing

            except Exception as e:
                print(f"从中间文件 {intermediate_filepath} 加载时出错: {e}。将尝试重新处理。")
                # traceback.print_exc()
                dataset = None # Mark for reprocessing

        # --- If intermediate file doesn't exist, failed loading, or force_reprocess is True ---
        if dataset is None:
            print(f"\n正在从原始文件处理并转换为 Xarray: 土地利用 '{land_use}'...")

            # Step 1: Use joblib to process profiles in parallel
            print(f"  并行处理 {len(profile_list)} 个剖面 (使用 {N_JOBS if N_JOBS != -1 else '所有'} 核)...")
            # Instantiate the Parallel context manager
            # backend='loky' is often more robust for complex tasks than 'multiprocessing'
            parallel_executor = Parallel(n_jobs=N_JOBS, backend='loky', verbose=0) # Set verbose=5 or 10 for progress updates

            # Create the list of delayed tasks for merge_profile_data
            tasks = [
                delayed(merge_profile_data)(project_dir, profile, land_use, group_list, mapping_df)
                for profile in profile_list
            ]

            # Execute the tasks in parallel
            # Wrap in try/except to catch potential errors during parallel execution
            try:
                results = parallel_executor(tasks)
                 # Check results immediately
                num_successful = sum(1 for res in results if res is not None and not res.empty)
                print(f"  并行处理完成。成功合并 {num_successful} / {len(profile_list)} 个剖面。")

            except Exception as e:
                 print(f"并行处理剖面时发生严重错误 for land_use '{land_use}': {e}")
                 # traceback.print_exc()
                 results = [None] * len(profile_list) # Assume failure for all if Parallel fails globally

            # Reconstruct the profile_dataframes dictionary from the results
            profile_dataframes = {}
            total_points_collected = 0
            for profile, merged_df in zip(profile_list, results):
                if merged_df is not None and not merged_df.empty:
                    profile_dataframes[profile] = merged_df
                    total_points_collected += merged_df.shape[1]
                # else: warnings should be handled within merge_profile_data or if parallel execution failed

            print(f"  收集到 {len(profile_dataframes)} 个剖面的数据，总计 {total_points_collected} 个有效观测点序列。")

            # Step 2: Convert collected Pandas DataFrames to one Xarray Dataset
            if profile_dataframes:
                print(f"\n  正在将 {len(profile_dataframes)} 个剖面的合并数据转换为 Xarray Dataset...")
                # Pass land_use for better logging inside the function
                dataset = convert_to_xarray_dataset(profile_dataframes, mapping_df, land_use)

                if dataset is not None:
                    all_data[land_use] = dataset
                    # --- Save to intermediate file (Xarray Dataset) ---
                    try:
                        print(f"  处理完成，正在保存到中间文件: {intermediate_filepath}")
                        # Ensure any previous file handle is closed if overwriting
                        if os.path.exists(intermediate_filepath):
                             try:
                                 # Attempt to close dataset if it was loaded before failing validation
                                 if intermediate_filepath in xr.backends.file_manager.FILE_CACHE:
                                     xr.backends.file_manager.FILE_CACHE.pop(intermediate_filepath).close()
                             except Exception as close_err:
                                 print(f"    尝试关闭旧文件句柄时出现非致命错误: {close_err}")

                        # Use to_netcdf for Xarray Dataset saving
                        write_job = dataset.to_netcdf(intermediate_filepath, mode='w', engine='netcdf4', compute=False) # Use compute=False for dask-backed arrays if any, here it might not matter much but good practice
                        write_job.compute() # Trigger the actual writing
                        print("  保存成功。")
                    except Exception as e:
                        print(f"保存中间文件 {intermediate_filepath} 时出错: {e}")
                        # traceback.print_exc()
                        # Should we keep the in-memory dataset even if saving failed? Yes.
                else:
                    print(f"错误: 未能为土地利用 '{land_use}' 生成有效的 Xarray Dataset。")
                    # all_data[land_use] remains None or non-existent

            else: # No valid profile dataframes were collected after parallel processing
                print(f"错误: 对于土地利用 '{land_use}', 并行处理后未能收集到任何剖面的有效原始数据。请检查原始 Obs.out 文件和映射。")
                # Ensure land_use key doesn't exist or is None
                if land_use in all_data: del all_data[land_use]

        # Optional: Close the dataset if it's no longer needed in memory right away,
        # especially in a loop processing many land uses. But keep it in all_data dict.
        # if dataset is not None and land_use in all_data:
        #     dataset.close() # Close file handle associated with the dataset if loaded from disk


    print("\n数据加载及合并完成。")
    # Final check on loaded data
    loaded_counts = {lu: (ds[TARGET_VAR].shape if ds else None) for lu, ds in all_data.items()}
    print("最终加载的数据集:")
    for lu, shape in loaded_counts.items():
        print(f"  - {lu}: {'加载成功 Shape='+str(shape) if shape else '加载失败或无数据'}")

    return all_data


# --- 函数：提取特定时间点的所有数据点 (从 Xarray Dataset) ---
# (No changes needed here, works with the Xarray dataset produced)
def get_data_at_time_xarray(all_data, land_use, time_point_sim_day):
    """
    从 Xarray Dataset 中提取指定土地利用在特定模拟天数时间点的所有观测点数据。
    返回一个DataFrame，包含 PhysicalX, PhysicalY, PhysicalZ, WaterContent, Profile, Group, ObsPointID 等坐标。
    """
    if land_use not in all_data or all_data[land_use] is None:
         print(f"错误: 未找到 '{land_use}' 土地利用的 Xarray 数据集。")
         return pd.DataFrame()

    dataset = all_data[land_use]

    # Check if dataset seems valid before proceeding
    if TARGET_VAR not in dataset or 'sim_day' not in dataset.coords or 'point' not in dataset.dims:
        print(f"错误: 土地利用 '{land_use}' 的数据集结构无效 (缺少变量/坐标/维度)。")
        return pd.DataFrame()

    # Ensure sim_day coordinate is loaded if using dask
    try:
         dataset['sim_day'].load()
    except Exception as load_err:
         print(f"错误: 加载 'sim_day' 坐标时出错 for {land_use}: {load_err}")
         return pd.DataFrame()


    # --- 提取时间切片 ---
    try:
        # Use Xarray's sel method with 'nearest'
        # Add tolerance if exact integer match is needed but float precision is tricky
        # However, 'nearest' is generally robust for this.
        time_slice_ds = dataset.sel(sim_day=time_point_sim_day, method='nearest')

        # Get the actual simulation day found
        actual_sim_day_used = time_slice_ds['sim_day'].values.item() # .item() extracts scalar

        # Check how close the nearest match is
        if abs(actual_sim_day_used - time_point_sim_day) > 0.5: # More than half a day away
             print(f"警告: 找到的最接近模拟日 {actual_sim_day_used:.2f} 与请求的 {time_point_sim_day} 相差超过半天。")


    except KeyError:
        # This might happen if sim_day isn't a coordinate or the dataset is empty
        print(f"错误: 无法使用 'sim_day' 选择时间切片。检查数据集坐标。")
        if 'sim_day' in dataset.coords:
             min_day = dataset['sim_day'].min().values
             max_day = dataset['sim_day'].max().values
             print(f"数据模拟天数范围: {min_day:.2f} 到 {max_day:.2f}")
        else:
             print("数据集中缺少 'sim_day' 坐标。")
        return pd.DataFrame()
    except Exception as e:
        print(f"提取时间切片时发生错误 for sim_day {time_point_sim_day}: {e}")
        # traceback.print_exc()
        return pd.DataFrame()

    # --- 将 Xarray 时间切片转换为 Pandas DataFrame ---
    try:
        print(f"  正在从时间切片 (实际模拟日 {actual_sim_day_used:.2f}) 提取数据到 Pandas DataFrame...")
        # Get the target variable values for all points at this time
        water_content_values = time_slice_ds[TARGET_VAR].values # This should be 1D array of values for all points

        # Get all coordinate values associated with the 'point' dimension
        point_coord_data = {}
        for coord_name in time_slice_ds.coords:
             if 'point' in time_slice_ds[coord_name].dims:
                 point_coord_data[coord_name] = time_slice_ds[coord_name].values

        # Combine water content and coordinate data into a DataFrame
        all_points_data_df = pd.DataFrame(point_coord_data)

        # Check if water_content_values length matches the number of points
        if len(water_content_values) != len(all_points_data_df):
             print(f"错误: 含水量数据点数量 ({len(water_content_values)}) 与坐标点数量 ({len(all_points_data_df)}) 不匹配！")
             return pd.DataFrame()

        all_points_data_df['WaterContent'] = water_content_values
        all_points_data_df['ActualTimeSimDay'] = actual_sim_day_used # Add actual sim day used

        # Filter out potential NaN water content resulting from outer joins or missing data
        initial_count = len(all_points_data_df)
        all_points_data_df = all_points_data_df.dropna(subset=['WaterContent'])
        final_count = len(all_points_data_df)
        if initial_count > final_count:
             print(f"  注意: 移除了 {initial_count - final_count} 个含水量为 NaN 的点。")


    except Exception as e:
        print(f"将 Xarray 时间切片转换为 Pandas DataFrame 时出错: {e}")
        # traceback.print_exc()
        return pd.DataFrame()


    if all_points_data_df.empty:
        print(f"错误: 在土地利用 '{land_use}', 模拟天数 {actual_sim_day_used:.2f} 没有找到可以用于可视化的有效观测点数据。")
        return pd.DataFrame()

    print(f"  成功提取 {len(all_points_data_df)} 个观测点的数据。")
    return all_points_data_df


# --- 函数：可视化单个剖面在特定时间的含水量分布 (二维，支持筛选) ---
# (No changes needed here)
def visualize_profile_water_content_2d(all_points_data_at_time_df, land_use, profile, actual_sim_day, wc_threshold=None, filter_type='below'):
    """
    可视化指定土地利用、剖面在特定时间点的含水量分布 (二维)，支持按含水量筛选。
    wc_threshold: 含水量阈值。
    filter_type: 'below' (低于阈值), 'above' (高于阈值), None (不筛选)。
    """
    if not isinstance(all_points_data_at_time_df, pd.DataFrame) or all_points_data_at_time_df.empty:
         print(f"错误: visualize_profile_water_content_2d 收到无效的输入 DataFrame。")
         return

    # Ensure 'Profile' column exists
    if 'Profile' not in all_points_data_at_time_df.columns:
        print(f"错误: 输入的 DataFrame 缺少 'Profile' 列，无法筛选剖面。")
        return

    profile_points_df = all_points_data_at_time_df[all_points_data_at_time_df['Profile'] == profile].copy()

    if profile_points_df.empty:
        print(f"信息: 在土地利用 '{land_use}', 剖面 '{profile}', 模拟天数 {actual_sim_day:.2f} 没有找到数据用于二维剖面可视化。")
        return

    # Check required columns for plotting
    required_plot_cols = ['PhysicalColumn', 'PhysicalDepth', 'WaterContent']
    if not all(col in profile_points_df.columns for col in required_plot_cols):
         print(f"错误: DataFrame for profile '{profile}' 缺少绘图所需的列 ({required_plot_cols})。可用列: {profile_points_df.columns.tolist()}")
         return

    x = profile_points_df['PhysicalColumn']
    y = profile_points_df['PhysicalDepth']
    z = profile_points_df['WaterContent']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color maps and labels based on filter type
    highlight_cmap = 'viridis'
    other_cmap = 'coolwarm' # Or 'Greys', 'binary_r' etc for less emphasis
    highlight_label = ''
    other_label = 'Other Points'
    highlight_alpha = 1.0
    other_alpha = 0.5
    highlight_size = 150
    other_size = 80

    points_to_highlight = pd.DataFrame()
    points_other = profile_points_df.copy() # Start with all points as 'other'


    if wc_threshold is not None and filter_type in ['below', 'above']:
        if filter_type == 'below':
            points_to_highlight = profile_points_df[profile_points_df['WaterContent'] <= wc_threshold]
            points_other = profile_points_df[profile_points_df['WaterContent'] > wc_threshold]
            highlight_label = f'{TARGET_VAR} <= {wc_threshold:.3f}'
            other_label = f'{TARGET_VAR} > {wc_threshold:.3f}'
            highlight_cmap = 'Blues_r' # Colder colors for lower moisture
            other_cmap = 'YlOrRd'    # Warmer colors for higher moisture
        else: # filter_type == 'above'
            points_to_highlight = profile_points_df[profile_points_df['WaterContent'] >= wc_threshold]
            points_other = profile_points_df[profile_points_df['WaterContent'] < wc_threshold]
            highlight_label = f'{TARGET_VAR} >= {wc_threshold:.3f}'
            other_label = f'{TARGET_VAR} < {wc_threshold:.3f}'
            highlight_cmap = 'YlOrRd' # Warmer colors for higher moisture
            other_cmap = 'Blues_r'    # Colder colors for lower moisture

        # Determine overall colorbar limits from *all* points on the profile
        vmin = profile_points_df['WaterContent'].min()
        vmax = profile_points_df['WaterContent'].max()
        norm = plt.Normalize(vmin, vmax)

        # Plot 'other' points first (as background)
        if not points_other.empty:
             ax.scatter(points_other['PhysicalColumn'], points_other['PhysicalDepth'],
                        c=points_other['WaterContent'], cmap=other_cmap, s=other_size, norm=norm,
                        edgecolors='grey', alpha=other_alpha, label=other_label)

        # Plot highlighted points on top
        if not points_to_highlight.empty:
             scatter_highlight = ax.scatter(points_to_highlight['PhysicalColumn'], points_to_highlight['PhysicalDepth'],
                                        c=points_to_highlight['WaterContent'], cmap=highlight_cmap, s=highlight_size, norm=norm,
                                        edgecolors='black', alpha=highlight_alpha, label=highlight_label)

             # Add a single colorbar representing the full range using a neutral map like viridis
             sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
             sm.set_array([]) # Needed for the colorbar
             cbar = fig.colorbar(sm, ax=ax, aspect=30)
             cbar.set_label(f'{TARGET_VAR} (Volumetric Water Content)')
             # Add legend for the highlight/other distinction
             ax.legend(title=f'Filter: {filter_type.capitalize()} {wc_threshold:.3f}')

        elif not points_other.empty: # Only 'other' points plotted, still add colorbar/legend
             sm = plt.cm.ScalarMappable(cmap=other_cmap, norm=norm)
             sm.set_array([])
             cbar = fig.colorbar(sm, ax=ax, aspect=30)
             cbar.set_label(f'{TARGET_VAR} (Volumetric Water Content)')
             ax.legend(title=f'Filter applied, no points matched highlight criteria')

        else: # No points left after filtering? Should not happen if profile_points_df was not empty initially
             ax.text(0.5, 0.5, 'No data points match filter criteria', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


    else: # No filtering, plot all points with a single colormap
        vmin = z.min()
        vmax = z.max()
        norm = plt.Normalize(vmin, vmax)
        scatter_all = ax.scatter(x, y, c=z, cmap='viridis', s=100, edgecolors='k', alpha=0.7, norm=norm)
        cbar = fig.colorbar(scatter_all, ax=ax, aspect=30)
        cbar.set_label(f'{TARGET_VAR} (Volumetric Water Content)')

    # Common plot settings
    ax.set_xlabel('Horizontal Position (PhysicalColumn)')
    ax.set_ylabel('Depth (PhysicalDepth, m)') # Assuming Z is depth
    try:
        actual_date = sim_day_to_date(actual_sim_day).strftime('%Y-%m-%d')
    except AttributeError: # Handle NaT date
        actual_date = "Unknown Date"

    ax.set_title(f'2D {TARGET_VAR} Profile: {land_use} - Profile {profile} at {actual_date} (Day {actual_sim_day:.0f})')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.invert_yaxis() # Depth increases downwards

    # Improve x-axis ticks if PhysicalColumn are discrete indices
    unique_x = np.unique(x)
    if len(unique_x) < 20 and pd.api.types.is_integer_dtype(x):
         ax.set_xticks(unique_x)
         ax.set_xticklabels([str(int(tick)) for tick in unique_x]) # Ensure labels are integers

    plt.tight_layout()
    plt.show()


# --- 函数：可视化整个区域在特定时间的含水量分布 (三维，支持筛选) ---
# (No changes needed here)
def visualize_water_content_3d(all_points_data_at_time_df, land_use, actual_sim_day, wc_threshold=None, filter_type='below'):
    """
    可视化指定土地利用在特定时间点的所有观测点含水量分布 (三维)，支持按含水量筛选。
    wc_threshold: 含水量阈值。
    filter_type: 'below' (低于阈值), 'above' (高于阈值), None (不筛选)。
    """
    if not isinstance(all_points_data_at_time_df, pd.DataFrame) or all_points_data_at_time_df.empty:
        print(f"错误: visualize_water_content_3d 收到无效的输入 DataFrame。")
        return

    # Check required columns for plotting
    required_plot_cols = ['PhysicalX', 'PhysicalY', 'PhysicalZ', 'WaterContent']
    if not all(col in all_points_data_at_time_df.columns for col in required_plot_cols):
         print(f"错误: 输入的 DataFrame 缺少 3D 绘图所需的列 ({required_plot_cols})。可用列: {all_points_data_at_time_df.columns.tolist()}")
         return


    x = all_points_data_at_time_df['PhysicalX']
    y = all_points_data_at_time_df['PhysicalY']
    z = all_points_data_at_time_df['PhysicalZ']
    c = all_points_data_at_time_df['WaterContent'] # Original water content for coloring

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define color maps and properties
    highlight_cmap = 'viridis'
    other_cmap = 'coolwarm'
    highlight_label = '' # Labels are harder in 3D scatter legends for color
    other_label = 'Other Points'
    highlight_alpha = 1.0
    other_alpha = 0.4 # Make others more transparent in 3D
    highlight_size = 100
    other_size = 30

    points_to_highlight = pd.DataFrame()
    points_other = all_points_data_at_time_df.copy()

    # Determine overall colorbar limits from *all* points
    vmin = c.min() if not c.empty else 0
    vmax = c.max() if not c.empty else 1
    norm = plt.Normalize(vmin, vmax)

    if wc_threshold is not None and filter_type in ['below', 'above']:
        filter_text = ""
        if filter_type == 'below':
            points_to_highlight = all_points_data_at_time_df[all_points_data_at_time_df['WaterContent'] <= wc_threshold]
            points_other = all_points_data_at_time_df[all_points_data_at_time_df['WaterContent'] > wc_threshold]
            highlight_cmap = 'Blues_r'
            other_cmap = 'YlOrRd'
            filter_text = f'(Highlight: WC <= {wc_threshold:.3f})'
        else: # filter_type == 'above'
            points_to_highlight = all_points_data_at_time_df[all_points_data_at_time_df['WaterContent'] >= wc_threshold]
            points_other = all_points_data_at_time_df[all_points_data_at_time_df['WaterContent'] < wc_threshold]
            highlight_cmap = 'YlOrRd'
            other_cmap = 'Blues_r'
            filter_text = f'(Highlight: WC >= {wc_threshold:.3f})'

        # Plot 'other' points first
        if not points_other.empty:
             ax.scatter(points_other['PhysicalX'], points_other['PhysicalY'], points_other['PhysicalZ'],
                        c=points_other['WaterContent'], cmap=other_cmap, marker='o', s=other_size, norm=norm,
                        alpha=other_alpha, edgecolors=None) # No edges for background points

        # Plot highlighted points
        scatter_highlight = None
        if not points_to_highlight.empty:
             scatter_highlight = ax.scatter(points_to_highlight['PhysicalX'], points_to_highlight['PhysicalY'], points_to_highlight['PhysicalZ'],
                                        c=points_to_highlight['WaterContent'], cmap=highlight_cmap, marker='o', s=highlight_size, norm=norm,
                                        edgecolors='black', alpha=highlight_alpha, depthshade=True) # Enable depthshade

        # Add a single colorbar representing the full range using a neutral map
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label(f'{TARGET_VAR} (Volumetric Water Content)')

        # Add filter info to title or annotation
        title_suffix = f"\n{filter_text}"

    else: # No filtering
        scatter_all = ax.scatter(x, y, z, c=c, cmap='viridis', marker='o', s=50, norm=norm,
                                 edgecolors='k', alpha=0.7, depthshade=True)
        cbar = fig.colorbar(scatter_all, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label(f'{TARGET_VAR} (Volumetric Water Content)')
        title_suffix = ""


    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)') # Assuming Z is vertical position, not necessarily depth

    try:
        actual_date = sim_day_to_date(actual_sim_day).strftime('%Y-%m-%d')
    except AttributeError:
        actual_date = "Unknown Date"

    ax.set_title(f'3D {TARGET_VAR} Distribution: {land_use} at {actual_date} (Day {actual_sim_day:.0f}){title_suffix}')

    # Consider inverting Z axis if it represents depth
    # ax.invert_zaxis()

    # Adjust view angle for better visibility
    ax.view_init(elev=25., azim=-65)

    plt.tight_layout()
    plt.show()


# --- 函数：可视化映射文件中的点 (三维验证) ---
# (No changes needed here)
def verify_mapping_file_3d(mapping_df):
    """
    可视化观测点映射文件中定义的所有点在三维空间中的位置。
    """
    if mapping_df is None or mapping_df.empty:
        print("映射文件数据为空或无效，无法进行验证可视化。")
        return

    # Check required columns
    required_map_cols = ['PhysicalX', 'PhysicalY', 'PhysicalZ', 'Profile']
    if not all(col in mapping_df.columns for col in required_map_cols):
        print(f"错误: 映射文件 DataFrame 缺少绘图所需的列 ({required_map_cols})。可用列: {mapping_df.columns.tolist()}")
        return


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_profiles = sorted(mapping_df['Profile'].unique()) # Sort profiles for consistent coloring
    # Use a perceptually uniform colormap like 'viridis' or 'plasma'
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_profiles)))
    profile_color_map = dict(zip(unique_profiles, colors))

    for profile in unique_profiles:
        profile_df = mapping_df[mapping_df['Profile'] == profile]
        ax.scatter(profile_df['PhysicalX'], profile_df['PhysicalY'], profile_df['PhysicalZ'],
                   color=profile_color_map[profile], label=f'Profile {profile}', s=50, alpha=0.8)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Visualization of Observation Points from Mapping File')

    # Optional: Invert Z if it represents depth
    # ax.invert_zaxis()

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # Move legend outside plot
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()


# --- 主执行流程 ---
# (Adjusted to include Joblib info and improved user interaction)
if __name__ == "__main__":
    print(f"--- Hydrus 3D Data Processor ---")
    print(f"Project Root: {PROJECT_ROOT_DIR}")
    print(f"Mapping File: {OBS_POINT_MAPPING_FILE}")
    print(f"Using {N_JOBS if N_JOBS != -1 else 'all available'} CPU cores for parallel processing.")

    # 1. Load observation point mapping file
    obs_mapping_df = load_obs_point_mapping(OBS_POINT_MAPPING_FILE)
    if obs_mapping_df is None:
        print("\n错误: 无法继续，观测点映射文件加载失败或不完整。请检查文件和路径。")
        exit() # Exit if mapping is essential and failed

    # Provide initial options
    print("\n--- 主菜单 ---")
    print("1. 验证观测点映射文件 (3D 可视化)")
    print("2. 加载/处理模拟数据并进入分析/可视化")
    print("q. 退出")

    while True:
        initial_choice = input("请输入选项 (1, 2, q): ").strip().lower()
        if initial_choice == '1':
            print("\n--- 验证映射文件 ---")
            verify_mapping_file_3d(obs_mapping_df)
            # Return to main menu after verification
            print("\n--- 主菜单 ---")
            print("1. 验证观测点映射文件 (3D 可视化)")
            print("2. 加载/处理模拟数据并进入分析/可视化")
            print("q. 退出")
        elif initial_choice == '2':
            break # Proceed to data loading
        elif initial_choice in ['q', 'quit', 'exit']:
            print("程序结束。")
            exit()
        else:
            print("输入无效，请输入 1, 2, 或 q。")

    # 2. Load/Process Simulation Data
    print("\n--- 数据加载/处理 ---")
    reprocess_input = input("是否强制重新处理所有模拟数据 (忽略缓存)? (yes/no, 默认 no): ").strip().lower()
    force_reprocess_flag = (reprocess_input == 'yes')

    # Call the optimized function using Xarray intermediate files and Joblib
    # This might take some time if reprocessing is needed
    all_merged_data_xarray = load_all_simulation_data_optimized_xarray(
         PROJECT_ROOT_DIR,
         PROFILE_LOCATIONS,
         OBS_GROUPS,
         LAND_USE_TYPES,
         obs_mapping_df,
         intermediate_dir=INTERMEDIATE_DATA_DIR,
         force_reprocess=force_reprocess_flag
    )

    # Check if any data was successfully loaded
    available_land_uses_with_data = [lu for lu, ds in all_merged_data_xarray.items() if ds is not None]

    if not available_land_uses_with_data:
         print("\n错误: 未能加载或处理任何有效的模拟数据。请检查:")
         print(f"  - 项目根目录 '{PROJECT_ROOT_DIR}' 是否正确且包含模拟文件夹。")
         print(f"  - 模拟文件夹名称是否符合预期格式 (例如 'DATE-PROFILE-GROUP-LANDUSE')。")
         print(f"  - '{OBS_OUT_FILENAME}' 文件是否存在于模拟文件夹中且格式正确。")
         print(f"  - 映射文件 '{OBS_POINT_MAPPING_FILE}' 是否与模拟数据匹配。")
         print("程序无法继续。")
         exit()
    else:
        print(f"\n成功加载/处理了以下土地利用类型的数据: {', '.join(available_land_uses_with_data)}")

    # 3. Interactive Visualization/Analysis Loop
    print("\n--- 可视化/分析模式 ---")
    while True:
        print("\n--- 分析选项 ---")
        print("1. 2D 剖面视图")
        print("2. 3D 整体视图")
        # Add future analysis options here
        print("q. 退出可视化/分析")

        viz_type = input("请输入选项 (1, 2, q): ").strip().lower()

        if viz_type in ['q', 'quit', 'exit']:
            break # Exit analysis loop
        elif viz_type not in ['1', '2']:
            print("输入无效，请重新输入。")
            continue

        # Select Land Use
        print("\n可用土地利用类型:", ", ".join(available_land_uses_with_data))
        selected_land_use = input("请输入要查看的土地利用类型: ").strip()

        if selected_land_use not in available_land_uses_with_data:
            print("输入无效或该土地利用类型无数据，请从可用列表中选择。")
            continue

        # Select Time Point (Date)
        dataset_for_selection = all_merged_data_xarray[selected_land_use]
        if dataset_for_selection is None or 'sim_day' not in dataset_for_selection.coords:
             print(f"错误: 土地利用 '{selected_land_use}' 的数据集无效或缺少模拟天数坐标。")
             continue

        try:
            # Ensure coordinates are loaded for min/max
            dataset_for_selection['sim_day'].load()
            min_sim_day = dataset_for_selection['sim_day'].min().item()
            max_sim_day = dataset_for_selection['sim_day'].max().item()

            start_date_dt = sim_day_to_date(min_sim_day)
            end_date_dt = sim_day_to_date(max_sim_day)

            start_date_prompt = start_date_dt.strftime('%Y-%m-%d') if not pd.isna(start_date_dt) else "N/A"
            end_date_prompt = end_date_dt.strftime('%Y-%m-%d') if not pd.isna(end_date_dt) else "N/A"

            # Suggest a date roughly in the middle
            suggested_sim_day = int((min_sim_day + max_sim_day) / 2) if not np.isnan(min_sim_day) and not np.isnan(max_sim_day) else (min_sim_day if not np.isnan(min_sim_day) else 0)
            suggested_date_dt = sim_day_to_date(suggested_sim_day)
            suggested_date_prompt = suggested_date_dt.strftime('%Y-%m-%d') if not pd.isna(suggested_date_dt) else start_date_prompt

            print(f"\n数据时间范围: {start_date_prompt} (Day {min_sim_day:.0f}) 到 {end_date_prompt} (Day {max_sim_day:.0f})")
            date_input = input(f"请输入要查看的日期 (YYYY-MM-DD 格式，例如 {suggested_date_prompt}): ").strip()

            selected_sim_day = date_to_sim_day(date_input)
            if selected_sim_day is None:
                 continue # Error message already printed by date_to_sim_day
            if not (min_sim_day <= selected_sim_day <= max_sim_day):
                 print(f"警告: 输入日期 {date_input} (Day {selected_sim_day}) 超出实际数据范围 [{min_sim_day:.0f}, {max_sim_day:.0f}]。将查找最近的可用日期。")


        except Exception as e:
            print(f"获取或处理时间范围时出错: {e}")
            continue


        # Extract data for the selected time using the Xarray function
        print(f"\n正在提取 '{selected_land_use}' 在日期 {date_input} (模拟日 {selected_sim_day}) 附近的数据...")
        all_points_data_at_selected_time_df = get_data_at_time_xarray(
             all_merged_data_xarray,
             selected_land_use,
             selected_sim_day
         )

        if all_points_data_at_selected_time_df.empty:
             print(f"无法进行可视化，在日期 {date_input} (或附近) 未找到有效的 '{selected_land_use}' 观测点数据。")
             continue # Go back to analysis options loop

        # Get the actual simulation day used by 'nearest' selection
        actual_sim_day_used = all_points_data_at_selected_time_df['ActualTimeSimDay'].iloc[0]
        actual_date_used_dt = sim_day_to_date(actual_sim_day_used)
        actual_date_used_str = actual_date_used_dt.strftime('%Y-%m-%d') if not pd.isna(actual_date_used_dt) else f"Sim Day {actual_sim_day_used:.0f}"
        print(f"  (实际使用的数据来自: {actual_date_used_str})")


        # --- Ask for filtering options ---
        filter_threshold = None
        filter_type = None
        while True: # Loop for filter input validation
            filter_input = input(f"是否按 {TARGET_VAR} 值进行筛选突出显示? (yes/no, 默认 no): ").strip().lower()
            if filter_input == 'yes':
                 try:
                     threshold_input = input(f"  请输入 {TARGET_VAR} 阈值 (例如 0.15): ").strip()
                     filter_threshold = float(threshold_input)

                     type_input = input(f"  突出显示低于 ('below') 或高于 ('above') 阈值? (默认 below): ").strip().lower()
                     if type_input == 'above':
                         filter_type = 'above'
                     else:
                         filter_type = 'below' # Default
                         if type_input not in ['below', '']: print("  输入无效，默认为 'below'。")

                     print(f"  将突出显示 {filter_type} 阈值 {filter_threshold:.3f} 的区域。")
                     break # Valid filter input
                 except ValueError:
                     print("  阈值输入无效，请输入一个数值。")
                     # Loop again to ask for filtering
            elif filter_input in ['no', '']:
                 print("  不进行筛选。")
                 break # No filter selected
            else:
                 print("  输入无效，请输入 'yes' 或 'no'。")


        # --- Perform Visualization ---
        if viz_type == '1': # 2D Profile View
            # Find available profiles within the extracted DataFrame for this time step
            if 'Profile' not in all_points_data_at_selected_time_df.columns:
                 print("错误: 提取的数据中缺少 'Profile' 列，无法进行剖面可视化。")
                 continue

            profiles_at_this_time = sorted(all_points_data_at_selected_time_df['Profile'].unique())

            if not profiles_at_this_time:
                 print(f"错误: 在时间点 {actual_date_used_str}，土地利用 '{selected_land_use}' 下没有找到任何剖面数据点。")
                 continue

            print(f"\n在该时间点有数据的剖面: {', '.join(profiles_at_this_time)}")
            selected_profile = input("请输入要查看的剖面位置 (例如 '15'): ").strip()

            if selected_profile not in profiles_at_this_time:
                print("输入无效或该剖面在该时间点无数据，请从可用列表中选择。")
                continue

            # Call 2D visualization
            visualize_profile_water_content_2d(
                all_points_data_at_selected_time_df,
                selected_land_use,
                selected_profile,
                actual_sim_day_used, # Pass the actual sim day used
                wc_threshold=filter_threshold,
                filter_type=filter_type
            )

        elif viz_type == '2': # 3D Volume View
             # Call 3D visualization
             visualize_water_content_3d(
                 all_points_data_at_selected_time_df,
                 selected_land_use,
                 actual_sim_day_used, # Pass the actual sim day used
                 wc_threshold=filter_threshold,
                 filter_type=filter_type
             )

    # End of analysis loop
    print("\n退出可视化/分析模式。")
    print("程序结束。")
