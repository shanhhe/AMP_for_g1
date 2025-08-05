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

MOTION_FILES = glob.glob('datasets/customed_g1/fist.csv')
# MOTION_FILES = glob.glob('datasets/joint_from_simulation/forward_1.0.csv')  # Replace with your actual path to the motion files
CARTESIAN_MOTION_FILES = glob.glob('datasets/cartesian_with_orientation_from_simulation/forward_1.0.csv')  # Replace with your actual path to the motion files
CARTESIAN_AND_JOINT_MOTION_FILES = glob.glob('datasets/joints_and_cartesian_from_simulation/forward_1.0.csv')


class G129AMPCfg( G1LeggedRobotCfg ):

    class env( G1LeggedRobotCfg.env ):
        num_actions = 29
        num_envs = 2048
        include_history_steps = None  # Number of steps of history to include.
        # 3 + 3 + 3 + 3 + 21 + 21 + 21 + 2 = 77
        # num_observations = 74
        # num_privileged_obs = 77
        num_observations = 96 #original amp
        num_privileged_obs = 99
        reference_state_initialization = True
        reference_state_initialization_prob = 1
        amp_motion_files = MOTION_FILES
        episode_length_s = 20 # episode length in seconds
        g1_cartesian_link_names = ["left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",  
                              "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link",
                                "torso_link", "head_link",
                              "left_shoulder_roll_link", "left_elbow_link", "left_rubber_hand",
                              "right_shoulder_roll_link", "right_elbow_link", "right_rubber_hand"]
        key_point_names = ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"]
        data_type = 'joint'  # 'cartesian' or 'joint' or 'joints_and_cartesian'

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
            
            'waist_yaw_joint': 0.0,   # [rad]
            'waist_roll_joint': 0.0,   # [rad]
            'waist_pitch_joint': 0.0,   # [rad]

            'left_shoulder_pitch_joint': 0.0,   # [rad]
            'left_shoulder_roll_joint': 0.0,   # [rad]
            'left_shoulder_yaw_joint': 0.0,   # [rad]
            'left_elbow_joint': 0.0,   # [rad]
            'left_wrist_roll_joint': 0.0,   # [rad]
            'left_wrist_pitch_joint': 0.0,   # [rad]
            'left_wrist_yaw_joint': 0.0,   # [rad]

            'right_shoulder_pitch_joint': 0.0,   # [rad]
            'right_shoulder_roll_joint': 0.0,   # [rad]
            'right_shoulder_yaw_joint': 0.0,   # [rad]
            'right_elbow_joint': 0.0,   # [rad]
            'right_wrist_roll_joint': 0.0,   # [rad]
            'right_wrist_pitch_joint': 0.0,   # [rad]
            'right_wrist_yaw_joint': 0.0,   # [rad]
        }

    class control( G1LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'waist_yaw': 300,
                     'waist_roll': 90,
                     'waist_pitch': 80,
                     'knee': 150,
                     'ankle': 40,
                     'shoulder_pitch': 90,
                     'shoulder_roll': 60,
                     'shoulder_yaw': 20.,
                     'elbow': 60,
                     'wrist_roll': 10,
                     'wrist_pitch': 20,
                     'wrist_yaw': 20,
                    #  'ankle_pitch': 35,
                    #  'ankle_roll': 30,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'waist_yaw': 3,
                     'waist_roll': 0.8,
                     'waist_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder_pitch': 2,
                     'shoulder_roll': 1,
                     'shoulder_yaw': 0.4,
                     'elbow': 1,
                     'wrist_roll': 0.4,
                     'wrist_pitch': 0.2,
                     'wrist_yaw': 0.4,

 
 
                    #  'ankle_pitch': 4,
                    #  'ankle_roll': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( G1LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class asset( G1LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof.urdf'
        name = 'g1_amp'
        foot_name = "ankle_roll"
        penalize_contacts_on = ["head", "hip", "wrist", "torso", "shoulder", "elbow", "knee"]
        knee_name = "knee"
        # terminate_after_contacts_on = ["pelvis", "head", "hip", "wrist", "torso", "shoulder", "elbow", "knee"]
        terminate_after_contacts_on = ["pelvis", "head"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        terminate_after_base_z = 0.4
        # selected_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        #                             12, 13, 14,
        #                             15, 16, 18, 22, 23, 25]  # 指定你想保留的关节索引
        selected_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14,
                                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        end_effector_name = "right_wrist_yaw_link"
        
    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.0
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

            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = 1.0
            # base_height = -1.0
            strike = 20.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = -0.5
            action_rate = -0.01
            dof_pos_limits = -2.0
            # minimize_torso_angular_velocity = 2.0
            # minimize_waist_pitch_deviation = 1.0
            # minimize_waist_roll_deviation = 1.0
            # minimize_waist_yaw_deviation = 0.2
            # torso_yaw_smoothness = 0.2

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        linear_increasing_commands_for_play = False # if true: increase the linear velocity commands during play
        linear_decreasing_commands_for_play = False
        class ranges:
            # lin_vel_x = [-0.3, 1.2] # min max [m/s]
            # lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            # ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            lin_vel_x = [-0.9, 1.75] # min max [m/s]
            lin_vel_y = [-1.4, 0.58]   # min max [m/s]
            ang_vel_yaw = [-7, 4.4]    # min max [rad/s]
            heading = [-3.14, 3.14]
            target_radius = [0.8, 1.2] # min max [m]
            target_theta = [-1, 1] # min max [rad]
            target_z = 1.0 # min max [m]


    # class sim( G1LeggedRobotCfg.sim ):
    #     dt = 0.005

class G129AMPCfgPPO( G1LeggedRobotCfgPPO ):
    runner_class_name = 'G1AMPOnPolicyRunner'
    class algorithm( G1LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01 # 0.005 for cartesian, 0.015 for joint space
        # entropy_coef = 0.012 # 0.005 for cartesian, 0.015 for joint space
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4
        gamma = 0.99

        # # -- Random Network Distillation
        # class rnd_cfg: 
        #     weight = 100.0 # initial weight of the RND reward

        #     # note: This is a dictionary with a required key called "mode".
        #     #   Please check the RND module for more information.
        #     weight_schedule = None

        #     reward_normalization: False  # whether to normalize RND reward # not improve training
        #     state_normalization: True  # whether to normalize RND state observations # curiosity normalization(Recommended to be True)

        #     # -- Learning parameters
        #     learning_rate = 0.001  # learning rate for RND

        #     # -- Network parameters
        #     # note: if -1, then the network will use dimensions of the observation
        #     num_outputs = 1  # number of outputs of RND network
        #     predictor_hidden_dims = [16, 16] # hidden dimensions of predictor network
        #     target_hidden_dims = [16]  # hidden dimensions of target network

        # class symmetry_cfg:

        #     use_data_augmentation = True  # this adds symmetric trajectories to the batch
        #     use_mirror_loss = False  # this adds symmetry loss term to the loss function

        #     # string containing the module and function name to import.
        #     # Example: "legged_gym.envs.locomotion.anymal_c.symmetry:get_symmetric_states"
        #     #
        #     # .. code-block:: python
        #     #
        #     #     @torch.no_grad()
        #     #     def get_symmetric_states(
        #     #        obs: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None, cfg: "BaseEnvCfg" = None, obs_type: str = "policy"
        #     #     ) -> Tuple[torch.Tensor, torch.Tensor]:
        #     #
        #     data_augmentation_func = "legged_gym.envs.g1_21.g1_21_amp_env:data_augmentation_func_g1"

        #     # coefficient for symmetry loss term
        #     # if 0, then no symmetry loss is used
        #     mirror_loss_coeff = 0.0

    class runner( G1LeggedRobotCfgPPO.runner ):
        run_name = 'init_gamma0.99_addkeypos_change_taskreward'
        experiment_name = 'fight_new_task'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 20000 # number of policy updates
        empirical_normalization = True # whether to use empirical normalization of the observations
        save_interval = 100

        amp_reward_coef = 0.2
        amp_motion_files = MOTION_FILES
        amp_cartesian_motion_files = CARTESIAN_MOTION_FILES
        amp_cartesian_and_joint_motion_files = CARTESIAN_AND_JOINT_MOTION_FILES
        
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3 # smaller to rely more on style reward(imitation)
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.02] * 29

  