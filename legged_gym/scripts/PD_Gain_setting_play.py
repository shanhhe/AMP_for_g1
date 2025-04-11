import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import matplotlib.pyplot as plt


import numpy as np
import torch
import json
import pandas as pd


def update_camera_position(env, robot_index, camera_offset):
    """Update the camera position to track the robot."""
    # Get the robot's current position
    robot_position = env.root_states[robot_index, :3].cpu().numpy()

    # Calculate the new camera position
    new_camera_position = robot_position + camera_offset

    # Update the camera position and look-at target
    env.set_camera(new_camera_position, robot_position)

def _check_command_interface():
    """Check for command updates from the shared file."""
    try:
        with open('command_interface.json', 'r') as file:
            command_interface = json.load(file)
        return command_interface
    except FileNotFoundError:
        return {}

def _update_command_ranges(env, command_interface):
    """Update command ranges in the environment."""
    if 'lin_vel_x' in command_interface:
        env.command_ranges["lin_vel_x"] = command_interface['lin_vel_x']
    if 'lin_vel_y' in command_interface:
        env.command_ranges["lin_vel_y"] = command_interface['lin_vel_y']
    if 'ang_vel_yaw' in command_interface:
        env.command_ranges["ang_vel_yaw"] = command_interface['ang_vel_yaw']
    if 'heading' in command_interface:
        env.command_ranges["heading"] = command_interface['heading']



def stabilize_robot(env, joint_index):
    # Define the fixed root state for the robot
    # fixed_position = [0.0, 0.0, 1.5]  # x, y, z position
    # fixed_orientation = [0.0, 0.0, 0.0, 1.0]  # Quaternion: x, y, z, w
    # print('num_envs:',env.num_envs)
    # Create a tensor for the root states
    # root_states = torch.zeros((env.num_envs, 13), device=env.device, dtype=torch.float32)
    
    # Set position (indices 0, 1, 2)
    # root_states[:, 0:3] = torch.tensor(fixed_position, device=env.device)

    # Set orientation (indices 3, 4, 5, 6)
    # root_states[:, 3:7] = torch.tensor(fixed_orientation, device=env.device)

    default_dof_pose = env.default_dof_pos.transpose(0, 1)
    
    # Set the root state for all environments
    # env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root_states))



    base_handle = env.gym.get_actor_handle(env.envs[0], 0)

    _dof_states = env.gym.acquire_dof_state_tensor(env.sim)
    
    dof_states = gymtorch.wrap_tensor(_dof_states)

    num_dofs = env.gym.get_actor_dof_count(env.envs[0], base_handle)

    

    all_dof_states = dof_states.clone()

    
    # fixed_states = torch.
    
    env_index = 0
    num_envs = args.num_envs
    start_index = env_index * num_dofs
    end_index = start_index + num_dofs

    # print(dof_states[start_index:end_index])

    env_dof_states = dof_states[start_index:end_index, :]
    for dof_idx in range(num_dofs):
        if dof_idx != joint_index:
            env_dof_states[dof_idx, 0] = default_dof_pose[dof_idx]
            env_dof_states[dof_idx, 1] = 0.0
        # env_dof_states[dof_idx, 0] = default_dof_pose[dof_idx]
        # env_dof_states[dof_idx, 1] = 0.0
    all_dof_states[start_index:end_index, :] = env_dof_states
    actor_indices = torch.tensor([0], dtype=torch.int32, device=env.device)

    # prop = env.gym.get_actor_dof_properties(env.envs[0], 0)
    # for i in range(num_dofs):
    #     # if i != joint_index:
    #     print(prop['stiffness'])
    # print(all_dof_states)
    env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(all_dof_states), gymtorch.unwrap_tensor(actor_indices), 1)



    env.gym.refresh_dof_state_tensor(env.sim)

        
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    init_pos = [0.0, 0.0, 3.5]  # Initial position of the robot
    init_ori = [0.0, 0.0, 0.0, 1.0]  # Initial orientation of the robot
    env_cfg.init_state.pos = init_pos
    env_cfg.init_state.rot = init_ori
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.fix_base_link = True
    env_cfg.control.control_type = 'T'
    env_cfg.control.action_scale = 1.0
    # env_cfg.asset.disable_gravity = True

    env_cfg.env.test = True
    robot_index = 0  # Index of the robot to track
    stop_state_log = 100  # Number of steps for logging states

    camera_offset = np.array([2.0, 0.0, 1.0])  # Adjust this offset as needed

    # Prepare environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("env_cfg:",env_cfg)
    print("env:",env)
    # obs = env.get_observations()

    logger = Logger(env.dt)
    logger.log_dir = args.log_dir
    stop_rew_log = env.max_episode_length + 1  # Steps for average reward calculation



    # Load policy
    # train_cfg.runner.resume = True
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)

    # Export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)
    # Define the sine wave parameters
    amplitude = args.amplitude  # Amplitude of the sine wave
    frequency = args.frequency  # Frequency of the sine wave (Hz)
    time_step = env.dt  # Time step of the simulation
    time = 0  # Initialize time

    # joint_name = args.joint_name  # Index for waist_roll_joint
    # joint_index = env.dof_dict[joint_name]
    joint_index = 0
    joint_name = env.dof_names[joint_index]

    for i in range(10 * int(env.max_episode_length)):
        # Check and update command ranges
        # command_interface = _check_command_interface()
        # _update_command_ranges(env, command_interface)
        # actions = policy(obs.detach())
        # obs, _, rews, dones, infos = env.step(actions.detach())

        # Correct initialization of custom_actions
        custom_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

    
        

        # Custom PD tuning values (these can be tuned interactively or systematically)
        desired_positions = torch.zeros((env.num_envs, env.num_dofs), device=env.device)
        desired_velocities = torch.zeros_like(desired_positions)
        # Convert time to a torch.Tensor
        time_tensor = torch.tensor(time, device=env.device)

        # Compute the desired position as a sine wave
        desired_positions[:, [joint_index]] = amplitude * torch.sin(2 * np.pi * frequency * time_tensor) + args.constant
        # desired_positions[:, [joint_index]] = 0.5

        # Increment time
        time += time_step
        # p_gains = torch.tensor([100.0, 50], device=env.device)  # PD position gains for roll and pitch
        # d_gains = torch.tensor([6.0, 4.0], device=env.device)    # PD velocity gains for rol·l and pitch
        p_gains = torch.tensor([args.p_gain], device=env.device)  # PD position gains for roll and pitch
        d_gains = torch.tensor([args.d_gain], device=env.device)    # PD velocity gains for roll and pitch

        # Calculate torques for waist joints
        # joint_positions = env.dof_pos[:, [waist_roll_index, waist_pitch_index]]
        # joint_velocities = env.dof_vel[:, [waist_roll_index, waist_pitch_index]]

        joint_positions = env.dof_pos[:, [joint_index]]
        # print('joint_positions:',joint_positions)
        # print('joint_positions[:, 0]:',joint_positions[:, 0])
        joint_velocities = env.dof_vel[:, [joint_index]]
        # torques = p_gains * (desired_positions[:, [waist_roll_index, waist_pitch_index]] - joint_positions) \
        #         - d_gains * joint_velocities
        torques = p_gains * (desired_positions[:, joint_index] - joint_positions) \
                - d_gains * joint_velocities
        # print('torques:',torques)
        # Apply calculated torques
        # custom_actions[:, [waist_roll_index, waist_pitch_index]] = torques
        custom_actions[:, joint_index] = 0

        # Step the environment with custom actions
        obs, _, rews, dones, infos, _, _ = env.step(custom_actions)

        # # print('env.torques:',env.torques)
        # # Attach the robot to a fixed point in the simulation
        # base_handle = env.gym.get_actor_handle(env.envs[0], 0)

        # # Define a fixed root state
        # fixed_root_state = gymapi.Transform(
        #     p=gymapi.Vec3(0.0, 0.0, 1.0),  # Position above the ground
        #     r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # Neutral orientation
        # )

        # Apply the fixed root state
        # stabilize_robot(env, joint_index)


        # # Update the camera position dynamically
        if MOVE_CAMERA:
            update_camera_position(env, robot_index, camera_offset)

        # # Optional: Add logic for rendering or saving frames if needed
        # if RECORD_FRAMES and i % 2 == 0:
        #     filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'frames', f'{i}.png')
        #     env.gym.write_viewer_image_to_file(env.viewer, filename)

        # Logging states
        # if i < stop_state_log:
        #     logger.log_states(
        #     {
        #         # f'actual_torque_{args.joint_name}': env.torques[:, joint_index].cpu().numpy(),
        #         # f'torque_{args.joint_name}': torques[:, 0].cpu().numpy(),
        #         # # 'waist_pitch_torque': torques[:, 1].cpu().numpy(),
        #         # f'position_{args.joint_name}': joint_positions[:, 0].cpu().numpy(),
        #         # # 'waist_pitch_position': joint_positions[:, 1].cpu().numpy(),
        #         # f'velocity_{args.joint_name}': joint_velocities[:, 0].cpu().numpy(),
        #         # f'desired_position_{args.joint_name}': desired_positions[:, joint_index].cpu().numpy(),
        #         # f'error_{args.joint_name}': (desired_positions[:, joint_index] - joint_positions[:, 0]).cpu().numpy(),
        #         # f'p_gain_{args.joint_name}': p_gains.item(),
        #         # f'd_gain_{args.joint_name}': d_gains.item(),
        #         # 'waist_pitch_velocity': joint_velocities[:, 1].cpu().numpy(),

        #         f'actual_torque_{joint_name}': env.torques[:, joint_index].cpu().numpy(),
        #         f'torque_{joint_name}': torques[:, 0].cpu().numpy(),
        #         # 'waist_pitch_torque': torques[:, 1].cpu().numpy(),
        #         f'position_{joint_name}': joint_positions[:, 0].cpu().numpy(),
        #         # 'waist_pitch_position': joint_positions[:, 1].cpu().numpy(),
        #         f'velocity_{joint_name}': joint_velocities[:, 0].cpu().numpy(),
        #         f'desired_position_{joint_name}': desired_positions[:, joint_index].cpu().numpy(),
        #         f'error_{joint_name}': (desired_positions[:, joint_index] - joint_positions[:, 0]).cpu().numpy(),
        #         f'p_gain_{joint_name}': p_gains.item(),
        #         f'd_gain_{joint_name}': d_gains.item(),
        #     }
        # )
        # elif i == stop_state_log:
        #     logger.plot_states(p_gains.item(), d_gains.item())
        #     stop_state_log += 100
        #     stabilize_robot(env, joint_index)
        #     # joint_index += 1
        #     args.torque += 50
            # joint_name = env.dof_names[joint_index]
            
            # args.p_gain += 10
            # args.d_gain += 0.05
            # env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(desired_positions))
        #     # logger.plot_states()
        #     plot_logged_data(logger.state_log)

            

        # Logging rewards
        # if 0 < i < stop_rew_log and infos["episode"]:
        #     num_episodes = torch.sum(env.reset_buf).item()
        #     if num_episodes > 0:
        #         logger.log_rewards(infos["episode"], num_episodes)
        # elif i == stop_rew_log:
        #     logger.print_rewards()

    # # Save logged states
    # save_states_to_csv(logger.state_log, env.dt, '/home/shanhe/unitree_rl_gym/data/bruce/PD_Gain_setting_play')


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True

    args = get_args()
    args.task='g1_amp'
    args.num_envs = 1
    #{'ankle_pitch_l': 4, 'ankle_pitch_r': 9, 'hip_pitch_l': 1, 'hip_pitch_r': 6, 'hip_roll_l': 2, 'hip_roll_r': 7, 'hip_yaw_l': 0, 'hip_yaw_r': 5, 'knee_pitch_l': 3, 'knee_pitch_r': 8}
    args.joint_name = 'left_hip_roll_joint'
    args.torque = -500
    args.amplitude = 1  # Amplitude of the sine wave
    args.frequency = 0.5  # Frequency of the sine wave (Hz)
    args.constant = 0  # Constant offset for the sine wave
    args.log_dir = '/home/shanhe/AMP_for_hardware/legged_gym/data/g1_amp/PD_Gain_setting_play'
    args.p_gain = 100.0
    args.d_gain = 0

    play(args)
