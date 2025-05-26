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
import glob

from legged_gym.envs.base.g1_legged_robot_config import G1LeggedRobotCfg, G1LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/g1/walk1_subject1_.csv')


class G1LEGAMPCfg( G1LeggedRobotCfg ):

    class env( G1LeggedRobotCfg.env ):
        num_actions = 12
        num_envs = 2048
        include_history_steps = None  # Number of steps of history to include.
        # 3 + 3 + 3 + 3 + 12 + 12 + 12 + 2 = 50
        # num_observations = 47
        # num_privileged_obs = 50
        num_observations = 42 #original amp
        num_privileged_obs = 48
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class init_state( G1LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': -0.15,   # [rad]
            'left_hip_roll_joint': 0.0,   # [rad]
            'left_hip_yaw_joint': 0.0 ,  # [rad]
            'left_knee_joint': 0.3,   # [rad]
            'left_ankle_pitch_joint': -0.15,   # [rad]
            'left_ankle_roll_joint': 0.0,   # [rad]
            'right_hip_pitch_joint': -0.15,   # [rad]
            'right_hip_roll_joint': 0.0,   # [rad]
            'right_hip_yaw_joint': 0.0 ,  # [rad]
            'right_knee_joint': 0.3,   # [rad]
            'right_ankle_pitch_joint': -0.15,   # [rad]
            'right_ankle_roll_joint': 0.0,   # [rad]
        }

    class control( G1LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {   'hip_yaw': 100,
                        'hip_roll': 100,
                        'hip_pitch': 100,
                        'knee': 150,
                        'ankle': 40,
                     }  # [N*m/rad]
        damping = {     'hip_yaw': 2,
                        'hip_roll': 2,
                        'hip_pitch': 2,
                        'knee': 4,
                        'ankle': 2,
                     }     # [N*m*s/rad]
        # stiffness = {   'hip_yaw': 0,
        #                 'hip_roll': 0,
        #                 'hip_pitch': 0,
        #                 'knee': 0,
        #                 'ankle': 0,
        #              }  # [N*m/rad]
        # damping = {     'hip_yaw': 0,
        #                 'hip_roll': 0,
        #                 'hip_pitch': 0,
        #                 'knee': 0,
        #                 'ankle': 0,
        #              }     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6

    class terrain( G1LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class asset( G1LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = 'g1_amp'
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        knee_name = "knee"
        # terminate_after_contacts_on = ["pelvis", "head", "hip", "wrist", "torso", "shoulder", "elbow", "knee"]
        terminate_after_contacts_on = ["pelvis", "head", "hip", "wrist", "torso", "shoulder", "elbow", "knee"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        selected_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        terminate_after_base_z = 0.4
  
    class domain_rand:
        randomize_friction = False
        friction_range = [0.1, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 3.]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.5
        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards( G1LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        class scales( G1LeggedRobotCfg.rewards.scales ):
            # termination = 0.0
            # tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            # tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            # lin_vel_z = 0.0
            # ang_vel_xy = 0.0
            # orientation = 0.0
            # torques = 0.0
            # dof_vel = 0.0
            # dof_acc = 0.0
            # base_height = 0.0 
            # feet_air_time =  0.0
            # collision = 0.0
            # feet_stumble = 0.0
            # action_rate = 0.0
            # stand_still = 0.0
            # dof_pos_limits = 0.0

            # tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            # tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = -1.0
            # base_height = -10.0
            # dof_acc = -2.5e-7
            # dof_vel = -1e-3
            # feet_air_time = 0.0
            # collision = -0.1
            # action_rate = -0.01
            # dof_pos_limits = -5.0
            # alive = 0.15
            # hip_pos = -1.0
            # contact_no_vel = -0.2
            # feet_swing_height = -20.0
            # contact = 0.18
            # straight_stance_phase = 2.0
            # penalty_knee_hyperextension = 1.0
            # stance_swing_coordination = 2.0
            # swing_height = 0.5

            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            # alive = 0.15
            # hip_pos = -1.0
            # contact_no_vel = -0.2
            # feet_swing_height = -20.0
            # contact = 0.18

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 4.0] # min max [m/s]
            lin_vel_y = [-0.1, 0.1]   # min max [m/s]
            ang_vel_yaw = [-0.2, 0.2]    # min max [rad/s]
            heading = [-3.14, 3.14]

    # class sim( G1LeggedRobotCfg.sim ):
    #     dt = 0.005

class G1LEGAMPCfgPPO( G1LeggedRobotCfgPPO ):
    runner_class_name = 'G1AMPOnPolicyRunner'
    class algorithm( G1LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( G1LeggedRobotCfgPPO.runner ):
        run_name = 'init_1'
        experiment_name = 'g1_12_without_unitree'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 5000 # number of policy updates

        amp_reward_coef = 0.02
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3 # smaller to rely more on style reward(imitation)
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

  