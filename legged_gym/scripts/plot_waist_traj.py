import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from isaacgym.torch_utils import *
from pybullet_utils import transformations
from rsl_rl.datasets import pose3d
import torch

# —— 配置 —— #
file_name = 'walk3_backward'          # 替换为你的 CSV 文件名
file_path = f'datasets/walk/{file_name}.csv'         # 替换为你的 CSV 路径
fs = 30                     # 采样率 30 Hz
dt = 1 / fs                 # 采样间隔 1/30 秒

# 检查文件
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Not found: {file_path}")

df = pd.read_csv(file_path, header=None)

# —— 速度计算 —— #
# 第一列为 X 坐标
x = df.iloc[:, 0]
y = df.iloc[:, 1]
ang_vel = []

for i in range(1, df.shape[0]):
    quat_curr = torch.tensor(df.iloc[i, 3:7].values, dtype=torch.float32)
    quat_prev = torch.tensor(df.iloc[i-1, 3:7].values, dtype=torch.float32)
    quat_diff = transformations.quaternion_multiply(
                    quat_curr,
                    transformations.quaternion_inverse(quat_prev)
                )
    # 计算四元数的角度变化
     # Convert to axis-angle representation
    axis, angle = pose3d.QuaternionToAxisAngle(quat_diff)

    # Ensure angle is in the range [0, pi]
    if angle > np.pi:
        angle = 2 * np.pi - angle
        axis = -axis
    ang_vel.append(axis * angle / dt)
    print(f"Frame {i}: Angular Velocity = {ang_vel[-1]}")

# 转换 ang_vel (list of np.array) 为 N x 3 数组
ang_vel_array = np.stack(ang_vel)

# 计算相邻帧的时间间隔（长度 N-1）
# dt = np.diff(df.index) * dt  # 假设每行代表一帧，时间间隔为 dt
# print(f"数据长度 = {len(df)} 帧, 采样率 = {fs} Hz, 时间间隔 = {dt[0]} 秒， {len(dt)}")
# # 计算瞬时速度
# # 相邻帧的位移（长度 N-1）
# dx = x.diff().iloc[1:].to_numpy()
# vel_x = dx / dt
# dy = y.diff().iloc[1:].to_numpy()
# vel_y = dy / dt
# # 计算瞬时速度的模长（长度 N-1)
# v_magnitude = np.sqrt(dx**2 + dy**2) / dt
# # 将速度转换为 Pandas Series
# v = pd.Series(v_magnitude, index=df.index[1:])  # 从第二帧开始，因为第一帧没有前一帧
# # 计算最大和最小瞬时速度
# # 跳过 NaN 值
# v = v.dropna()  # 去除 NaN 值
# # 找到最大和最小速度的索引
# max_idx = v.idxmax()
# print(f"最大瞬时速度 = {v.max()}（单位/秒），发生在第 {max_idx} 帧")
# min_idx = v.idxmin()
# max_speed = v.loc[max_idx]
# min_speed = v.loc[min_idx]
# max_time = max_idx * 1 / 30
# min_time = min_idx * 1 / 30
# print(f"最大瞬时速度 = {max_speed}（单位/秒），发生在第 {max_idx} 帧，时间 = {max_time} 秒")
# print(f"最小瞬时速度 = {min_speed}（单位/秒），发生在第 {min_idx} 帧，时间 = {min_time} 秒")

# # 1. 拼装数据
# out_df = pd.DataFrame({
#     "vel_x": vel_x,   # 长度 N-1
#     "vel_y": vel_y,   # 长度 N-1
#     "ang_vel_z": ang_vel_array[:, 2],  # 长度 N-1
# })

# # 2. 可选：加索引，保证每列对齐（比如帧号，frame_id）
# out_df.index = df.index[1:]  # 从第1帧到最后一帧

# # 3. 保存为 CSV
# out_file = f"data/commands/{file_name}_velocity.csv"
# out_df.to_csv(out_file, index=False, header=False, float_format='%.6f')
# print(f"已保存至 {out_file}")


# ----------- Plot Linear Speed ----------- #
# plt.figure(figsize=(12, 4))
# plt.plot(v.index * dt[0], vel_x, label='Linear Speed dx', color='blue')
# plt.plot(v.index * dt[0], vel_y, label='Linear Speed dy', color='orange')
# plt.xlabel("Time (s)")
# plt.ylabel("Speed (units/s)")
# plt.title("Linear Speed over Time")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # ----------- Plot Angular Velocity ----------- #
# # 转换 ang_vel (list of np.array) 为 N x 3 数组
# ang_vel_array = np.stack(ang_vel)  # Shape: (N-1, 3)
# time_axis = np.arange(1, len(df)) * dt[0]  # 对应帧的时间点（从第1帧开始）

# plt.figure(figsize=(12, 6))
# plt.plot(time_axis, ang_vel_array[:, 0], label='ω_x')
# plt.plot(time_axis, ang_vel_array[:, 1], label='ω_y')
# plt.plot(time_axis, ang_vel_array[:, 2], label='ω_z')
# plt.xlabel("Time (s)")
# plt.ylabel("Angular Velocity (rad/s)")
# plt.title("Angular Velocity over Time")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # —— 角度轨迹 —— #
# # 计算关节角度轨迹
# # 假设关节角度在第 20, 21, 22 列（根据实际数据调整）
# cols = [df.iloc[:, i] for i in (19, 20, 21)]  # 关节角度列
# # 计算每个关节的瞬时速度

# 生成时间轴
N = int(len(df))
print(f"数据长度 = {N} 帧, 采样率 = {fs} Hz, 时间间隔 = {dt} 秒")
t = np.arange(N) * dt      # [0, 1/30, 2/30, …]


# cols = [df.iloc[:N, i] for i in (19, 20, 21)]

all_joint_names = ("left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint")  # 关节名称

selected_joint_names = ["left_wrist_pitch_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint"]  # 选择腰部关节名称

joint_names_index = [all_joint_names.index(name) + 7 for name in selected_joint_names]

cols = [df.iloc[:N, i] for i in joint_names_index]

n_rows = len(cols)

fig, axes = plt.subplots(n_rows, 1, figsize=(18, 12), sharex=True)
# for ax, col, joint_name in zip(axes, cols, ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")):
for ax, col, joint_name in zip(axes, cols, selected_joint_names):
    ax.plot(t, col)
    ax.set_ylabel(f'Angle')
    ax.set_title(f'{joint_name} trajectory')
    ax.grid(True)
axes[-1].set_xlabel('time (s)')
plt.xticks(np.arange(0, N * dt, 1), rotation=45)
plt.tight_layout()

# —— 保存图像 —— #
output_path = f'/home/shanhe/AMP_for_hardware/legged_gym/data/test/{file_name}_reference_{selected_joint_names[0]}.png'     # 输出文件名，可改成 .png/.jpg/.pdf 等
plt.savefig(output_path, dpi=300, bbox_inches='tight')
