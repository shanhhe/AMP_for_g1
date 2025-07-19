import os
import numpy as np
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from isaacgym.torch_utils import *
import torch
import cv2
import shutil

def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def angle_euclidean(x, y):
    # x, y 均为向量
    angle_diff = (x - y + np.pi) % (2 * np.pi) - np.pi
    return np.linalg.norm(angle_diff)

def update_camera_position(env, robot_index, camera_offset):
    """Update the camera position to track the robot."""
    # Get the robot's current position
    robot_position = env.root_states[robot_index, :3].cpu().numpy()

    # Calculate the new camera position
    new_camera_position = robot_position + camera_offset

    # Update the camera position and look-at target
    env.set_camera(new_camera_position, robot_position)

def get_amp_observations(env):
#     """ Get AMP observations
#     """
    N = env.num_envs
    K = len(env.cartesian_data_link_indices)

    # 基座四元数
    base_q = env.base_quat                  # (N,4)

    # 把 base_q expand 到 (N,K,4) 再展平
    base_q_exp = base_q.unsqueeze(1).expand(N, K, 4).reshape(N*K, 4)  # (N*K,4)
    base_q_inv = quat_conjugate(base_q_exp.reshape(N, K, 4))  # (N*K,4)

    # --- 1) link pos & vel world → local ---
    world_key_body_pose = env.rigid_body_states[:, env.cartesian_data_link_indices, 0:3]   # (N,K,3)
    world_key_body_quat = env.rigid_body_states[:, env.cartesian_data_link_indices, 3:7]  # (N,K,4)

    local_key_body_pos = world_key_body_pose - env.root_states[:, 0:3].unsqueeze(1)  # (N,K,3)
    local_link_quat = quat_mul(base_q_inv, world_key_body_quat) # (N,K,4)
    local_link_quat_flat = local_link_quat.reshape(-1, 4)    # (N*K, 4)
    roll, pitch, yaw = get_euler_xyz(local_link_quat_flat)
    # print(f"local_link_euler shape: {roll.shape}, {pitch.shape}, {yaw.shape}")
    
    roll = roll.view(N, K)
    pitch = pitch.view(N, K)
    yaw = yaw.view(N, K)

    # print(f"local_link_euler shape: {roll.shape}, {pitch.shape}, {yaw.shape}")

    local_link_euler = torch.stack([roll, pitch, yaw], dim=-1)  # (N*K, 3)
    # print(f"local_link_euler shape: {local_link_euler.shape}")

    # 展平
    flat_end_pos = local_key_body_pos.view(N*K, 3)            # (N*K,3)

    # 逆旋转到本地
    local_end_pos = quat_rotate_inverse(base_q_exp, flat_end_pos)  # (N*K,3)

    # 再恢复 (N, K*3)
    flat_local_key_pos = local_end_pos.view(N, K*3)
    flat_local_euler = local_link_euler.reshape(N, K * 3)  # (N,K*3)
    # print(f"flat_local_key_pos shape: {flat_local_key_pos.shape}, flat_local_euler shape: {flat_local_euler.shape}")

    # --- 2) 基座速度 world → local ---
    base_lin_vel = env.base_lin_vel   # (N,3)
    base_ang_vel = env.base_ang_vel   # (N,3)
    # print(f"base_lin_vel shape: {base_lin_vel.shape}, base_ang_vel shape: {base_ang_vel.shape}")

    joint_pos = env.dof_pos # 21

    return base_lin_vel.cpu().numpy().squeeze(), base_ang_vel.cpu().numpy().squeeze(), joint_pos.cpu().numpy().squeeze(), \
           flat_local_key_pos.cpu().numpy().squeeze(), flat_local_euler.cpu().numpy().squeeze()

def load_reference_motion_csv(csv_file):
    return np.loadtxt(csv_file, delimiter=',')

def play(args):
    # === 1. 环境初始化与 policy 加载 ===
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.reference_state_initialization = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.ranges.lin_vel_x = [1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.resampling_time = 3.5
    env_cfg.env.episode_length_s = 8
    train_cfg.runner.amp_num_preload_transitions = 10
    ref_csv = 'datasets/joints_and_cartesian_from_simulation/forward_1.0.csv'
    if args.data_type == 'cartesian':
        args.load_run = 'Jul16_13-21-37_walkvel1_cartesian_space_entropy_coef0.05_resume'
        env_cfg.env.data_type = 'cartesian'
    elif args.data_type == 'joints_and_cartesian':
        args.load_run = 'Jul16_00-42-19_walkvel1_cartesianandjoint_space_entropy_coef0.012'
        env_cfg.env.data_type = 'joints_and_cartesian'
    else:
        env_cfg.env.data_type = 'joint'  # 'cartesian' or 'joint' or 'joints_and_cartesian'
        args.load_run = 'Jul15_21-13-16_walkvel1_joint_space_entropy_coef0.012'

    robot_index = 0
    camera_offset = np.array([2.0, 0.0, 1.0])
    img_idx = 0
    video = None
    if RECORD_FRAMES:
        frames_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
        video_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'records')
        if not os.path.isdir(video_path):
            os.makedirs(video_path, exist_ok=True)
        if not os.path.isdir(frames_path):
            os.makedirs(frames_path, exist_ok=True)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs, _ = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # === 2. 加载 Reference AMP obs（CSV）===
    ref_csv_file = ref_csv if 'ref_csv' in locals() else "datasets/cartesian_with_orientation_from_simulation/forward_1.0.csv"
    ref_data = load_reference_motion_csv(ref_csv_file)  # shape (T, obs_dim)
    T = ref_data.shape[0]
    period_motion = [130, T]
    print(f"Loaded reference AMP obs from {ref_csv_file}, shape: {ref_data.shape}")

    ref_joint_space = ref_data[:, 7:28]  # Joint positions (21 joints)
    ref_base_lin_vel = ref_data[:, 28:31]  # Base linear velocity (3D)
    ref_base_ang_vel = ref_data[:, 31:34]  # Base angular velocity
    ref_cartesian_pos = ref_data[:, 55:88]  # Cartesian positions (11 links * 3D = 33 links)
    ref_cartesian_quat = ref_data[:, 88:]  # Cartesian orientations (11 links quaternion = 11 * 4 = 44)
    ref_cartesian_quat_flat = ref_cartesian_quat.reshape(-1, 4)
    ref_cartesian_quat_flat = torch.from_numpy(ref_cartesian_quat_flat).float()  # .float() 防止默认 double
    roll, pitch, yaw = get_euler_xyz(ref_cartesian_quat_flat)
    roll = roll.view(T, 11)
    pitch = pitch.view(T, 11)
    yaw = yaw.view(T, 11)
    ref_cartesian_euler = torch.stack([roll, pitch, yaw], dim=-1)  # (T, 11, 3)
    ref_cartesian_euler = ref_cartesian_euler.reshape(-1, 33).numpy()  # (T, 33)
    ref_cartesian_euler = np.unwrap(ref_cartesian_euler, axis=0)
    print(f"Reference cartesian euler shape: {ref_cartesian_euler.shape}")
    # print(f"Reference cartesian quat shape: {ref_cartesian_quat.shape}")

    # === 3. Policy rollout & MSE ===
    policy_joint_space = []
    policy_cartesian_space = []
    policy_base_lin_vel = []
    policy_base_ang_vel = []
    policy_cartesian_euler = []
    mse_per_frame = []

    obs, _ = env.reset()
    for t in range(T):
        with torch.no_grad():
            actions = policy(obs)
            obs, rews, dones, infos, _, _ = env.step(actions.detach())

        base_lin_vel, base_ang_vel, joint_pos, flat_local_key_pos, local_link_euler = get_amp_observations(env)
        policy_joint_space.append(joint_pos)
        policy_cartesian_space.append(flat_local_key_pos)
        policy_base_lin_vel.append(base_lin_vel)
        policy_base_ang_vel.append(base_ang_vel)
        policy_cartesian_euler.append(local_link_euler)
        mse = np.mean((joint_pos - ref_joint_space[t]) ** 2)
        mse_per_frame.append(mse)

        if FIXED_CAMERA:
            update_camera_position(env, robot_index, camera_offset)

        if RECORD_FRAMES:
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)
            if video is None:
                video = cv2.VideoWriter(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'records', f'{args.load_run}.mp4'), 
                                        cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1
    if RECORD_FRAMES:
        video.release()
        print(f"VideoWriter released, video file written, {img_idx} frames.")
        if os.path.isdir(frames_path):
            shutil.rmtree(frames_path)
            print(f"Deleted frames directory: {frames_path}")
    policy_joint_space = np.stack(policy_joint_space)
    policy_cartesian_space = np.stack(policy_cartesian_space)
    policy_base_lin_vel = np.stack(policy_base_lin_vel)
    policy_base_ang_vel = np.stack(policy_base_ang_vel)
    policy_cartesian_euler = np.stack(policy_cartesian_euler)
    policy_cartesian_euler = np.unwrap(policy_cartesian_euler, axis=0)
    print(f"policy_cartesian_euler shape: {policy_cartesian_euler.shape}")
    mse_per_frame = np.array(mse_per_frame)
    mean_mse = mse_per_frame.mean()

    ref_lin_x_error = np.mean((ref_base_lin_vel[period_motion[0]:period_motion[1], 0] - env.commands[:, 0].item()) ** 2)
    ref_lin_y_error = np.mean((ref_base_lin_vel[period_motion[0]:period_motion[1], 1] - env.commands[:, 1].item()) ** 2)
    ref_ang_yaw_error = np.mean((ref_base_ang_vel[period_motion[0]:period_motion[1], 2] - env.commands[:, 2].item()) ** 2)
    print(f"Current reference datatype: {env.cfg.env.data_type}")
    print(f"Ref Linear X MSE: {ref_lin_x_error:.6f}, Linear Y MSE: {ref_lin_y_error:.6f}, Angular Yaw MSE: {ref_ang_yaw_error:.6f}")

    policy_lin_x_error = np.mean((policy_base_lin_vel[period_motion[0]:period_motion[1], 0] - env.commands[:, 0].item()) ** 2)
    # policy_lin_x_error = np.mean(np.abs(policy_base_vel[:, 0] - env.commands[:, 0].item()))
    policy_lin_y_error = np.mean((policy_base_lin_vel[period_motion[0]:period_motion[1], 1] - env.commands[:, 1].item()) ** 2)
    policy_ang_yaw_error = np.mean((policy_base_ang_vel[period_motion[0]:period_motion[1], 2] - env.commands[:, 2].item()) ** 2)
    print(f"Policy Linear X MSE: {policy_lin_x_error:.6f}, Linear Y MSE: {policy_lin_y_error:.6f}, Angular Yaw MSE: {policy_ang_yaw_error:.6f}")

    
    # 假设你的原始数据如下（这里仅以第一个特征为例）
    ref_curve_joint_space = ref_joint_space[:, 4]   # (T,)
    policy_curve_joint_space = policy_joint_space[:, 4]
    ref_curve_cartesian_space = ref_cartesian_pos[:, 0]  # (T,)
    policy_curve_cartesian_space = policy_cartesian_space[:, 0]


    # 1. 找ref的峰，自动检测周期位置
    valleys_ref_joint_space, _ = find_peaks(-ref_curve_joint_space, distance=30)  # distance可调，越大越不会误判
    valleys_policy_joint_space, _ = find_peaks(-policy_curve_joint_space, distance=30)
    valleys_ref_cartesian_space, _ = find_peaks(-ref_curve_cartesian_space, distance=30)
    valleys_policy_cartesian_space, _ = find_peaks(-policy_curve_cartesian_space, distance=30)
    # print("Reference valleys at:", valleys_ref)
    # print("Policy valleys at:", valleys_policy)

    # 2. 选3个连续周期（假设峰之间为1周期）
    if len(valleys_ref_joint_space) < 4 or len(valleys_policy_joint_space) < 4:
        raise ValueError("ref_curve_joint_space中周期数太少,请采样更长轨迹")
    if len(valleys_ref_cartesian_space) < 4 or len(valleys_policy_cartesian_space) < 4:
        raise ValueError("ref_curve_cartesian_space中周期数太少,请采样更长轨迹")
    start_idx_ref = valleys_ref_joint_space[-5]
    end_idx_ref = valleys_ref_joint_space[-2]  # 取前3个完整周期
    start_idx_policy = valleys_policy_joint_space[-5]
    end_idx_policy = valleys_policy_joint_space[-2]

    # 3. 截取3周期的数据段
    ref_3period_joint_space = ref_joint_space[start_idx_ref:end_idx_ref, :]
    ref_3period_cartesian_space = ref_cartesian_pos[start_idx_ref:end_idx_ref, :]
    ref_3period_euler = ref_cartesian_euler[start_idx_ref:end_idx_ref, :]
    policy_3period_joint_space = policy_joint_space[start_idx_policy:end_idx_policy, :]  # 同步对齐（如不等长，用DTW对齐）
    policy_3period_cartesian_space = policy_cartesian_space[start_idx_policy:end_idx_policy, :]
    policy_3period_euler = policy_cartesian_euler[start_idx_policy:end_idx_policy, :]

    distance_joint_space, path_joint_space = fastdtw(policy_3period_joint_space, ref_3period_joint_space, dist=euclidean)
    distance_cartesian_space, path_cartesian_space = fastdtw(policy_3period_cartesian_space, ref_3period_cartesian_space, dist=euclidean)
    distance_cartesian_euler, path_cartesian_euler = fastdtw(policy_3period_euler, ref_3period_euler, dist=angle_euclidean)
    
    print(f"DTW distance joint space (3 periods): {distance_joint_space:.6f}")
    print(f"DTW distance per frame joint space (3 periods): {distance_joint_space / len(path_joint_space):.6f}")
    print(f"DTW distance per frame per joint space (3 periods): {distance_joint_space / (len(path_joint_space) * 21):.6f}")
    print(f"DTW distance cartesian space (3 periods): {distance_cartesian_space:.6f}")
    print(f"DTW distance per frame cartesian space (3 periods): {distance_cartesian_space / len(path_cartesian_space):.6f}")
    print(f"DTW distance per frame per link cartesian space (3 periods): {distance_cartesian_space / (len(path_cartesian_space) * 33):.6f}")
    print(f"DTW distance cartesian euler (3 periods): {distance_cartesian_euler:.6f}")
    print(f"DTW distance per frame cartesian euler (3 periods): {distance_cartesian_euler / len(path_cartesian_euler):.6f}")
    print(f"DTW distance per frame per link cartesian euler (3 periods): {distance_cartesian_euler / (len(path_cartesian_euler) * 33):.6f}")

    # 5. 可视化
    policy_aligned_joint_space = np.array([policy_3period_joint_space[i] for i, j in path_joint_space])
    policy_aligned_cartesian_space = np.array([policy_3period_cartesian_space[i] for i, j in path_cartesian_space])
    policy_aligned_cartesian_euler = np.array([policy_3period_euler[i] for i, j in path_cartesian_euler])
    ref_aligned_joint_space = np.array([ref_3period_joint_space[j] for i, j in path_joint_space])
    ref_aligned_cartesian_space = np.array([ref_3period_cartesian_space[j] for i, j in path_cartesian_space])
    ref_aligned_cartesian_euler = np.array([ref_3period_euler[j] for i, j in path_cartesian_euler])

    abs_err_per_joint = np.abs(policy_aligned_joint_space - ref_aligned_joint_space)
    print(f"policy_aligned_joint_space shape: {policy_aligned_joint_space.shape}")
    abs_err_per_link = np.abs(policy_aligned_cartesian_space - ref_aligned_cartesian_space)
    # joint_mae_after_alignment = np.sum(abs_err_per_joint)
    # link_mae_after_alignment = np.sum(abs_err_per_link)
    joint_mean_mae_after_alignment = np.mean(abs_err_per_joint)
    link_mean_mae_after_alignment = np.mean(abs_err_per_link)
    # print("Absolute error per joint after alignment:", joint_mae_after_alignment)
    print("Mean absolute error per joint after alignment:", joint_mean_mae_after_alignment)
    # print("Absolute error per link after alignment:", link_mae_after_alignment)  
    print("Mean absolute error per link after alignment:", link_mean_mae_after_alignment)

    abs_err_per_link_euler = np.abs(angle_diff(policy_aligned_cartesian_euler, ref_aligned_cartesian_euler))
    link_mean_mae_after_alignment_euler = np.mean(abs_err_per_link_euler)
    print("Mean absolute error per euler axis after alignment:", link_mean_mae_after_alignment_euler)

    # mean_mse_after_alignment = np.log(np.mean(mse_after_alignment) + 1e-8) # 防止log(0)导致-inf
    # mean_mse_after_alignment = np.mean(np.abs(mse_after_alignment))
    # print(f"Mean MSE after alignment: {mean_mse_after_alignment:.6f}")


    # print(f"\nMean AMP-observation MSE vs reference: {mean_mse:.5f}")

    # 你实验结束后统一写入csv
    results = {
        "run_name": args.load_run,
        "reference_datatype": args.data_type,
        "DTW_joint_space(3 periods)": f"{distance_joint_space:.5f}",
        "DTW_cartesian_space(3 periods)": f"{distance_cartesian_space:.5f}",
        "DTW_euler_axis(3 periods)": f"{distance_cartesian_euler:.5f}",
        "Joint_MAE": f"{joint_mean_mae_after_alignment:.5f}",
        "Link_MAE": f"{link_mean_mae_after_alignment:.5f}",
        "Euler_axis_MAE": f"{link_mean_mae_after_alignment_euler:.5f}",
        # "Ref Linear X MSE": f"{ref_lin_x_error:.5f}",
        # "Ref Linear Y MSE": f"{ref_lin_y_error:.5f}",
        # "Ref Angular Yaw MSE": f"{ref_ang_yaw_error:.5f}",
        "Policy Linear X MSE": f"{policy_lin_x_error:.5f}",
        "Policy Linear Y MSE": f"{policy_lin_y_error:.5f}",
        "Policy Angular Yaw MSE": f"{policy_ang_yaw_error:.5f}",
    }

    # 写入CSV   
    if WRITE_CSV:
        out_dir = "data/reference_compare"
        os.makedirs(out_dir, exist_ok=True)  # 如果目录不存在就创建

        out_file = os.path.join(out_dir, f"result.txt")

        with open(out_file, "a", newline='') as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
        # with open(csv_file, mode='w', newline='') as file:
        #     writer = csv.DictWriter(file, fieldnames=results.keys())
        #     if not file_exists:
        #         writer.writeheader()
        #     writer.writerow(results)
        print(f"Results saved to {out_file}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    k = 0
    axs[0].plot(policy_aligned_cartesian_euler[:, k], label='Policy joint[4] obs (aligned)')
    axs[0].plot(ref_aligned_cartesian_euler[:, k], '--', label='Reference joint[4] obs (aligned)')
    axs[0].set_title(f'DTW Aligned (3 Periods): Policy vs Reference (joint[4] obs) ref_data({args.data_type})')
    axs[0].legend(loc='upper left')

    # # 右侧y轴: MSE
    # ax0_right = axs[0].twinx()
    # ax0_right.plot(abs_err_per_link[:, 7], label='MSE (aligned)', color='red', linewidth=1)
    # # ax0_right.set_ylim(0, 0.005)
    # ax0_right.set_ylabel('MSE')
    # ax0_right.legend(loc='upper right')

    axs[1].plot(policy_cartesian_euler[:, k], label='Policy joint[4] obs', alpha=0.5)
    axs[1].plot(ref_cartesian_euler[:, k], '--', label='Reference joint[4] obs', alpha=0.5)
    # axs[1].plot(mse_per_frame, label='Per-frame jointobs MSE', color='red')

    axs[1].set_title(f'Per-frame joint[4]-observation ref_data({args.data_type})')
    axs[1].legend()
    plt.savefig("your_figure.png", dpi=150)
    # plt.show()
    # for i in range(33):
    #     plt.figure()
    #     plt.plot(policy_cartesian_euler[:, i], label=f'ref_cartesian_euler {i}')
    #     plt.plot(ref_cartesian_euler[:, i], label=f'ref_cartesian_euler {i}')
    #     plt.legend()
    #     plt.savefig(f"your_figure_{i}.png", dpi=150)

    

if __name__ == '__main__':
    args = get_args()
    WRITE_CSV = True  # 是否写入CSV
    RECORD_FRAMES = False
    FIXED_CAMERA = False
    # 你可以支持 --ref_csv xxx.csv
    if not hasattr(args, 'ref_csv'):
        args.task = 'g1_21'
        args.data_type = 'joints_and_cartesian'  # 'cartesian' or 'joint' or 'joints_and_cartesian'
    play(args)