import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from isaacgym.torch_utils import *
import pandas as pd
import numpy as np

def update_camera_position(env, robot_index, camera_offset):
    """Update the camera position to track the robot."""
    # Get the robot's current position
    robot_position = env.root_states[robot_index, :3].cpu().numpy()

    # Calculate the new camera position
    new_camera_position = robot_position + camera_offset

    # Update the camera position and look-at target
    env.set_camera(new_camera_position, robot_position)

def get_commands():
    motion_file = 'data/commands/fight2_velocity.csv'
    commands = pd.read_csv(motion_file, header=None).values
    return commands

def play(args):
    # commands = get_commands()
    # print(f"Commands shape: {commands.shape}, first command: {commands[0]}")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs + 1, 1)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.reference_state_initialization = False
    env_cfg.commands.ranges.lin_vel_x =  [0.0, 0.0]  # range of linear velocity in x direction
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    env_cfg.commands.ranges.target_radius = [1.0, 5.0]  # range of target point radius
    env_cfg.commands.ranges.target_theta = [-1, 1]
    env_cfg.commands.ranges.target_z = [1.2, 1.2]  # target point z coordinate
    env_cfg.commands.resampling_time = 3
    env_cfg.env.debug_viz = True  # disable debug visualization
    log = False
   
    # env_cfg.commands.linear_increasing_commands_for_play = True
    # env_cfg.commands.increasing_scale = 0.5
    env_cfg.env.episode_length_s = 20
    train_cfg.runner.amp_num_preload_transitions = 10

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    stop_state_log = env.max_episode_length - 2# number of steps before plotting states
    camera_offset = np.array([2.0, 0.0, 1.0])  # Adjust this offset as needed
    N = env.num_envs
    K = len(env.cartesian_data_link_indices)  # number of key bodies for logging
    outputs_cartesian = []
    outputs_joint = []
    outputs_joints_and_cartesian = []
    outputs_cartesian_withorientation = []

    for i in range(int(env.max_episode_length)):
        # k = i // 5 % commands.shape[0]
        # # print(f"Step {i}, command: {commands[k]}")
        # env.command_ranges["lin_vel_x"] =  [commands[k, 0], commands[k, 0]]
        # env.command_ranges["lin_vel_y"] = [commands[k, 1], commands[k, 1]]
        # env.command_ranges["ang_vel_yaw"] = [commands[k, 2], commands[k, 2]]
        update_camera_position(env, robot_index, camera_offset)
        # if i >= int(2 / env.dt):
        actions = policy(obs.detach())
        obs, rews, dones, infos, _, _ = env.step(actions.detach())

        base_q = env.base_quat
        base_pos = env.root_states[:, 0:3]  # (N,3)
        base_lin_vel = env.base_lin_vel   # (N,3)
        base_ang_vel = env.base_ang_vel   # (N,3)

        base_q_exp = base_q.unsqueeze(1).expand(N, K, 4).reshape(N*K, 4)  # (N*K,4)
        base_q_inv = quat_conjugate(base_q_exp.reshape(N, K, 4))  # (N*K,4)

        world_key_body_pose = env.rigid_body_states[:, env.cartesian_data_link_indices, 0:3]   # (N,K,3)
        world_key_body_quat = env.rigid_body_states[:, env.cartesian_data_link_indices, 3:7]  # (N,K,4)

        local_key_body_pos = world_key_body_pose - env.root_states[:, 0:3].unsqueeze(1)  # (N,K,3)
        local_key_body_pos = local_key_body_pos.view(N, K*3)  # (N,K*3)
        local_link_quat = quat_mul(base_q_inv, world_key_body_quat) # (N,K,4) 

        flat_local_link_quat = local_link_quat.reshape(N, K * 4)  # (N,K*4)
        
        # conver to numpy array for logging
        local_key_body_pos = local_key_body_pos.cpu().numpy().squeeze(0)
        flat_local_link_quat = flat_local_link_quat.cpu().numpy().squeeze(0)
        base_lin_vel = base_lin_vel.cpu().numpy().squeeze(0)
        base_ang_vel = base_ang_vel.cpu().numpy().squeeze(0)
        base_pos = base_pos.cpu().numpy().squeeze(0)
        base_q = base_q.cpu().numpy().squeeze(0)

        dof_pos = env.dof_pos
        dof_vel = env.dof_vel
        dof_pos = dof_pos.cpu().numpy().squeeze(0)
        dof_vel = dof_vel.cpu().numpy().squeeze(0)

        out_cartesian = np.concatenate([base_pos, base_q, local_key_body_pos, base_lin_vel, base_ang_vel])
        out_cartesian_withorientation = np.concatenate([base_pos, base_q, local_key_body_pos, base_lin_vel, base_ang_vel])
        out_joint = np.concatenate([base_pos, base_q, dof_pos, base_lin_vel, base_ang_vel, dof_vel])
        out_joints_and_cartesian = np.concatenate([out_joint, local_key_body_pos])
        outputs_cartesian.append(out_cartesian)
        outputs_joint.append(out_joint)
        outputs_joints_and_cartesian.append(out_joints_and_cartesian)
        outputs_cartesian_withorientation.append(out_cartesian_withorientation)
        if i < stop_state_log:
            logger.log_states(
                {
                    "left_rubber_hand_x": local_key_body_pos[0],
                    "left_rubber_hand_y": local_key_body_pos[1],
                    "left_rubber_hand_z": local_key_body_pos[2],
                    "right_rubber_hand_x": local_key_body_pos[3],
                    "right_rubber_hand_y": local_key_body_pos[4],
                    "right_rubber_hand_z": local_key_body_pos[5],
                    "left_ankle_roll_link_x": local_key_body_pos[6],
                    "left_ankle_roll_link_y": local_key_body_pos[7],
                    "left_ankle_roll_link_z": local_key_body_pos[8],
                    "right_ankle_roll_link_x": local_key_body_pos[9],
                    "right_ankle_roll_link_y": local_key_body_pos[10],
                    "right_ankle_roll_link_z": local_key_body_pos[11],
                }
            )
        if i==stop_state_log and log:
            result = np.vstack(outputs_cartesian)
            out_cartesian_file = 'datasets/cartesian_from_simulation/forward_' + str(env_cfg.commands.ranges.lin_vel_x[0]) + '.csv'
            parent = os.path.dirname(out_cartesian_file)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok =True)
            np.savetxt(out_cartesian_file, result, delimiter=',', comments='', fmt='%.6f')
            print(f"Output saved to {out_cartesian_file}")


            out_joint_file = 'datasets/joint_from_simulation/forward_' + str(env_cfg.commands.ranges.lin_vel_x[0]) + '.csv'
            result = np.vstack(outputs_joint)
            parent = os.path.dirname(out_joint_file)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok =True)
            np.savetxt(out_joint_file, result, delimiter=',', comments='', fmt='%.6f')
            print(f"Output saved to {out_joint_file}")

            out_joints_and_cartesian_file = 'datasets/joints_and_cartesian_from_simulation/forward_' + str(env_cfg.commands.ranges.lin_vel_x[0]) + '.csv'
            result = np.vstack(outputs_joints_and_cartesian)
            parent = os.path.dirname(out_joints_and_cartesian_file)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok =True)
            np.savetxt(out_joints_and_cartesian_file, result, delimiter=',', comments='', fmt='%.6f')
            print(f"Output saved to {out_joints_and_cartesian_file}")
            
            out_cartesian_withorientation_file = 'datasets/cartesian_with_orientation_from_simulation/forward_' + str(env_cfg.commands.ranges.lin_vel_x[0]) + '.csv'
            result = np.vstack(outputs_cartesian_withorientation)
            parent = os.path.dirname(out_cartesian_withorientation_file)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok =True)
            np.savetxt(out_cartesian_withorientation_file, result, delimiter=',', comments='', fmt='%.6f')
            print(f"Output saved to {out_cartesian_withorientation_file}")
        if i == stop_state_log and args.plot:
            logger.plot_single_state(key=['left_rubber_hand_x', 'left_rubber_hand_y', 'left_rubber_hand_z'], run_name=args.load_run)

if __name__ == '__main__':
    args = get_args()
    args.plot = False  # whether to plot the state
    play(args)
