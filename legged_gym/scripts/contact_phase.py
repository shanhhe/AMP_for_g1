import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==== 1. Demo 数据（替换成你的数据即可） ====
t            = np.linspace(0, 55, 551)        # 100 Hz
left_contact  = (np.sin(0.4*t) > 0).astype(int)
right_contact = (np.sin(0.4*t + np.pi + 1) > 0).astype(int)
v_cmd        = 0.5*np.floor(t/5)
v_act        = v_cmd + 0.2*np.random.randn(len(t))

# ==== 2. contact → phase（关键修改） ====
# 先创建全 -1 的矩阵，用 -1 当“空白/透明”
# 方式 A：直接用 NaN 当空位  (最简洁)
phase_disp = np.full((4, len(t)), np.nan)            # 全 NaN

for k,(l,r) in enumerate(zip(left_contact, right_contact)):
    if l==1 and r==1:     phase_disp[0,k] = 0   # Double → 0
    elif l==0 and r==1:   phase_disp[1,k] = 1   # Right  → 1
    elif l==1 and r==0:   phase_disp[2,k] = 2   # Left   → 2
    else:                 phase_disp[3,k] = 3   # Flight → 3

# ==== 3. 画图 ====
fig, axes = plt.subplots(2, 1, figsize=(12,5), sharex=True,
                         gridspec_kw={'height_ratios':[2,3]})

# --- 3a 支撑相 ---
cmap = ListedColormap(['black','red','blue','teal'])   # 0,1,2,3 分别对应四种颜色
# 让 -1 不显示（透明）
cmap.set_bad(color='white')

im = axes[0].imshow(phase_disp, aspect='auto', origin='lower',
                    extent=[t[0], t[-1], 0, 4],
                    cmap=cmap, vmin=0, vmax=3)
axes[0].set_yticks(np.arange(0.5,4.5))
axes[0].set_yticklabels(['Double','Right','Left','Flight'])
axes[0].set_ylabel('Support Phase')
axes[0].set_xlim(t[0], t[-1]) 

# --- 3b 速度 ---
axes[1].step(t, v_cmd, where='post', label='Commanded', linewidth=2)
axes[1].plot(t, v_act, 'r', label='Actual', linewidth=1)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Base Velocity X (m/s)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()


# ==== 4. 保存图像（可选） ====
fig.savefig('contact_phase.png', dpi=300, bbox_inches='tight')
