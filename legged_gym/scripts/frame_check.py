import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rsl_rl.datasets import pose3d


# —— 配置 —— #
file_name = 'sprint'          # 替换为你的 CSV 文件名
file_path = f'datasets/customed_g1/{file_name}.csv'         # 替换为你的 CSV 路径
fs = 30                     # 采样率 30 Hz
dt = 1 / fs                 # 采样间隔 1/30 秒

# 检查文件
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Not found: {file_path}")

df = pd.read_csv(file_path, header=None)


pos = df.iloc[:, :3].to_numpy()
print(pos.shape)
dx = np.diff(pos[:, 0]); dy = np.diff(pos[:, 1])
heading_vel = np.arctan2(dy, dx)

q = df.iloc[:, 3:7].to_numpy()
qw, qx, qy, qz = q[:-1].T
yaw_q = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

corr = np.corrcoef(heading_vel, yaw_q)[0, 1]
print("corr = ", corr)

yaw = np.rad2deg(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
print("yaw range:", yaw.min(), yaw.max())
# yaw_raw: 直接由四元数解出的 yaw（弧度）
yaw_raw = np.unwrap(yaw_q)                 # 展开避免跳变
# 速度方向
heading_vel = np.arctan2(dy, dx)

corr_unwrapped = np.corrcoef(heading_vel, yaw_raw[:-1])[0,1]
print("corr after unwrap =", corr_unwrapped)