# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import cv2
import shutil

def update_camera_position(env, robot_index, camera_offset):
    """Update the camera position to track the robot."""
    # Get the robot's current position
    robot_position = env.root_states[robot_index, :3].cpu().numpy()

    # Calculate the new camera position
    new_camera_position = robot_position + camera_offset

    # Update the camera position and look-at target
    env.set_camera(new_camera_position, robot_position)

def _extract_upper_body_angular_velocity(env):
    """
    Extract angular velocities (roll rate, pitch rate, yaw rate) for the pelvis,
    waist_roll_link, and torso_link individually.

    Returns:
        tuple: Summed absolute angular velocities and individual angular velocities
            for pelvis, waist, and torso.
    """

    # Extract indices for each body part separately
    pelvis_idx = env.body_names.index('pelvis')
    waist_idx = env.body_names.index('waist_roll_link')
    torso_idx = env.body_names.index('torso_link')

    # Extract angular velocity states for each body part
    rigid_body_state = env.gym.acquire_rigid_body_state_tensor(env.sim)
    env.rigid_body_states = isaacgym.gymtorch.wrap_tensor(rigid_body_state)
    env.rigid_body_states_view = env.rigid_body_states.view(env.num_envs, -1, 13)
    env.feet_state = env.rigid_body_states_view[:, env.feet_indices, :]

    pelvis_state = env.rigid_body_states_view[:, pelvis_idx, :]
    waist_state = env.rigid_body_states_view[:, waist_idx, :]
    torso_state = env.rigid_body_states_view[:, torso_idx, :]

    # Extract angular velocity components (indices 10:13)
    pelvis_ang_vel = pelvis_state[:, 10:13]  # Shape: [num_envs, 3]
    waist_ang_vel = waist_state[:, 10:13]    # Shape: [num_envs, 3]
    torso_ang_vel = torso_state[:, 10:13]    # Shape: [num_envs, 3]

    # Compute the sum of absolute angular velocities
    upper_body_ang_vel_sum = torch.abs(pelvis_ang_vel) + torch.abs(waist_ang_vel) + torch.abs(torso_ang_vel)

    return upper_body_ang_vel_sum, pelvis_ang_vel, waist_ang_vel, torso_ang_vel

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.reference_state_initialization = True
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.ranges.lin_vel_x =  [0.1, 0.1]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.5, 0.5]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    env_cfg.commands.resampling_time = 5.0
    # env_cfg.asset.fix_base_link = True  # Fix the base link to avoid falling
    # env_cfg.commands.linear_increasing_commands_for_play = True
    # env_cfg.commands.increasing_scale = 0.9
    # env_cfg.env.episode_length_s = 75
    env_cfg.env.episode_length_s = 10
    

    train_cfg.runner.amp_num_preload_transitions = 10

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = env.max_episode_length - 1 # number of steps before plotting states
    # stop_state_log = 10 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_offset = np.array([2.0, 0.0, 1.0])  # Adjust this offset as needed
    robot_position = env.root_states[robot_index, :3].cpu().numpy()
    camera_position = robot_position + camera_offset
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    video = None
    video_duration = env_cfg.env.episode_length_s # seconds
    num_frames = int(video_duration / env.dt)

    for i in range(int(env.max_episode_length)+10):
        if i == int(28 / env.dt) and env_cfg.commands.linear_increasing_commands_for_play:
            print("Starting to reverse the commands.")
            env_cfg.commands.linear_decreasing_commands_for_play = True

        actions = policy(obs.detach())
        obs, rews, dones, infos, _, _ = env.step(actions.detach())
        if RECORD_FRAMES and i < num_frames:
            frames_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
            video_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'records')
            if not os.path.isdir(video_path):
                os.mkdir(video_path)
            if not os.path.isdir(frames_path):
                os.mkdir(frames_path)
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)
            if video is None:
                video = cv2.VideoWriter(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'records', f'{args.load_run}.mp4'), 
                                        cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1
        if video is not None and i == num_frames:
            video.release()
            print(f"VideoWriter released, video file written, {num_frames} frames.")
            if os.path.isdir(frames_path):
                shutil.rmtree(frames_path)
                print(f"Deleted frames directory: {frames_path}")
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_direction)
        if FIXED_CAMERA:
            update_camera_position(env, robot_index, camera_offset)
        if i < stop_state_log:
            upper_body_ang_vel_sum, pelvis_ang_vel, waist_ang_vel, torso_ang_vel = _extract_upper_body_angular_velocity(env)
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'waist_yaw_joint': env.dof_pos[robot_index, env.dof_names.index("waist_yaw_joint")].item(),
                    'waist_roll_joint': env.dof_pos[robot_index, env.dof_names.index("waist_roll_joint")].item(),
                    'waist_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("waist_pitch_joint")].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'base_vel_roll': env.base_ang_vel[robot_index, 0].item(),
                    'base_vel_pitch': env.base_ang_vel[robot_index, 1].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'projected_gravity_x': env.projected_gravity[robot_index, 0].item(),
                    'projected_gravity_y': env.projected_gravity[robot_index, 1].item(),
                    'projected_gravity_z': env.projected_gravity[robot_index, 2].item(),
                    'plevis_ang_vel_roll': pelvis_ang_vel[:, 0].item(),
                    'plevis_ang_vel_pitch': pelvis_ang_vel[:, 1].item(),
                    'plevis_ang_vel_yaw': pelvis_ang_vel[:, 2].item(),
                    'waist_ang_vel_roll': waist_ang_vel[:, 0].item(),
                    'waist_ang_vel_pitch': waist_ang_vel[:, 1].item(),
                    'waist_ang_vel_yaw': waist_ang_vel[:, 2].item(),
                    'torso_ang_vel_roll': torso_ang_vel[:, 0].item(),
                    'torso_ang_vel_pitch': torso_ang_vel[:, 1].item(),
                    'torso_ang_vel_yaw': torso_ang_vel[:, 2].item(),
                    'left_hip_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("left_hip_pitch_joint")].item(),
                    'right_hip_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("right_hip_pitch_joint")].item(),
                    'left_hip_roll_joint': env.dof_pos[robot_index, env.dof_names.index("left_hip_roll_joint")].item(),
                    'right_hip_roll_joint': env.dof_pos[robot_index, env.dof_names.index("right_hip_roll_joint")].item(),
                    'left_hip_yaw_joint': env.dof_pos[robot_index, env.dof_names.index("left_hip_yaw_joint")].item(),
                    'right_hip_yaw_joint': env.dof_pos[robot_index, env.dof_names.index("right_hip_yaw_joint")].item(),
                    'left_knee_joint': env.dof_pos[robot_index, env.dof_names.index("left_knee_joint")].item(),
                    'right_knee_joint': env.dof_pos[robot_index, env.dof_names.index("right_knee_joint")].item(),
                    'left_ankle_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("left_ankle_pitch_joint")].item(),
                    'right_ankle_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("right_ankle_pitch_joint")].item(),
                    'left_ankle_roll_joint': env.dof_pos[robot_index, env.dof_names.index("left_ankle_roll_joint")].item(),
                    'right_ankle_roll_joint': env.dof_pos[robot_index, env.dof_names.index("right_ankle_roll_joint")].item(),
                    'left_shoulder_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("left_shoulder_pitch_joint")].item(),
                    'right_shoulder_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("right_shoulder_pitch_joint")].item(),
                    'left_shoulder_roll_joint': env.dof_pos[robot_index, env.dof_names.index("left_shoulder_roll_joint")].item(),
                    'right_shoulder_roll_joint': env.dof_pos[robot_index, env.dof_names.index("right_shoulder_roll_joint")].item(),
                    'left_elbow_joint': env.dof_pos[robot_index, env.dof_names.index("left_elbow_joint")].item(),
                    'right_elbow_joint': env.dof_pos[robot_index, env.dof_names.index("right_elbow_joint")].item(),
                    'left_wrist_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("left_wrist_pitch_joint")].item(),
                    'right_wrist_pitch_joint': env.dof_pos[robot_index, env.dof_names.index("right_wrist_pitch_joint")].item(),
                    'left_wrist_roll_joint': env.dof_pos[robot_index, env.dof_names.index("left_wrist_roll_joint")].item(),
                    'right_wrist_roll_joint': env.dof_pos[robot_index, env.dof_names.index("right_wrist_roll_joint")].item(),
                    'left_wrist_yaw_joint': env.dof_pos[robot_index, env.dof_names.index("left_wrist_yaw_joint")].item(),
                    'right_wrist_yaw_joint': env.dof_pos[robot_index, env.dof_names.index("right_wrist_yaw_joint")].item(),
                }
            )
        elif i==stop_state_log and args.plot_states:
                # logger.plot_waist_states(run_name=args.load_run)
                # logger.plot_states(run_name=args.load_run)
                logger.plot_single_state(key=['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint'], run_name=args.load_run)
                # logger.plot_contact_phase(run_name=args.load_run)
                print("Plotted states.")
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIXED_CAMERA = True
    args = get_args()
    args.plot_states = False  # Set to True to plot states
    args.real_time_factor = 1.0  # Set to 0 for no real-time factor
    play(args)
