
from legged_gym.envs.base.g1_legged_robot import G1LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G129AMPRobot(G1LeggedRobot):
    def compute_observations(self):
        """ Computes observations
        """
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    # self.commands[:, :3] * self.commands_scale,
                                    self.target_pos[:, :2] - self.root_states[:, :2],
                                    self.target_pos[:, 2].unsqueeze(1),
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        # Remove velocity observations from policy observation.
        if self.num_obs == self.num_privileged_obs - 3:
            self.obs_buf = self.privileged_obs_buf[:, 3:]
        else:
            self.obs_buf = torch.clone(self.privileged_obs_buf)


    def _reset_target_pos(self, env_ids=None):
        # env_ids: 要重置 target 的环境编号, 支持 batch
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # 采样 base/root 当前 xy 坐标
        base_pos = self.root_states[env_ids, :2]  # [N, 2]

        # 随机采样 target 的相对偏移（比如半径在 [0.5, 1.5] 米，角度 0-2pi）
        radius = torch.empty(len(env_ids), device=self.device).uniform_(self.cfg.commands.ranges.target_radius[0], self.cfg.commands.ranges.target_radius[1])
        theta = torch.empty(len(env_ids), device=self.device).uniform_(self.cfg.commands.ranges.target_theta[0], self.cfg.commands.ranges.target_theta[1])
        offset = torch.stack([
            radius * torch.cos(theta),
            radius * torch.sin(theta)
        ], dim=-1)  # [N, 2]

        # target 在世界坐标系下
        target_xy = base_pos + offset  # [N, 2]

        target_z = torch.full((len(env_ids), 1), self.cfg.commands.ranges.target_z, device=self.device)   # [N, 1]

        self.target_pos[env_ids] = torch.cat([target_xy, target_z], dim=-1)  # [N, 3]
        self.has_hit[env_ids] = False  # 重置 has_his 状态
        
    def _reward_strike(self):
        # 1. 位置和速度
        x_star = self.target_pos                 # [num_envs, 3]
        x_root = self.root_states[:, :3]         # [num_envs, 3]
        x_eff = self.rigid_body_states[:, self.end_effector_index, 0:3]                # [num_envs, 3]
        x_eff_dot = self.rigid_body_states[:, self.end_effector_index, 7:10]            # [num_envs, 3]
        base_vel = self.root_states[:, 7: 10]

        # 2. 单位向量
        d_star = x_star - x_root
        d_star_unit = d_star / (torch.norm(d_star, dim=1, keepdim=True) + 1e-8)

        # 3. 距离
        eff_dist = torch.norm(x_star - x_eff, dim=1)
        root_dist = torch.norm(x_star - x_root, dim=1)

        # 5. r_near
        dot_v = torch.sum(d_star_unit * x_eff_dot, dim=1)
        r_near = 0.2 * torch.exp(-2 * eff_dist**2) + 0.8 * torch.clamp((2/3) * dot_v, 0, 1)

        # 6. r_far
        v_star=1.0           # 目标方向速度
        dot_v_base = torch.sum(d_star_unit * base_vel, dim=1)
        vel_term = torch.clamp(v_star - dot_v_base, min=0)
        r_far = 0.7 * torch.exp(-0.5 * root_dist ** 2) + 0.3 * torch.exp(-vel_term ** 2)

        # 7. reward 逻辑
        reward = torch.where(
            self.has_hit,
            torch.ones_like(root_dist),
            torch.where(
                root_dist < 1.375,
                0.3 * r_near + 0.3,
                0.3 * r_far
            )
        )

        return reward

    def _reward_minimize_torso_angular_velocity(self):
        """
        Penalize large angular velocities in the upper body to stabilize motion.

        Returns:
            torch.Tensor: The reward value for minimizing angular velocity.
        """
        _, pelvis_ang_vel, waist_ang_vel, torso_ang_vel = self._extract_upper_body_angular_velocity()


        # Compute penalty for angular velocity magnitude
        angular_velocity_penalty = torch.norm(torso_ang_vel, dim=1)  # Shape: [num_envs]

        # Reward is inversely proportional to the penalty
        reward = torch.exp(-angular_velocity_penalty)  # Penalize high angular velocity
        return reward
    
    def _reward_minimize_waist_pitch_deviation(self, target_pitch=0.0, weight=1.0, log=False):
        """
        Penalize deviation of waist_pitch_joint from a target angle.

        Args:
            target_pitch (float): Desired waist pitch angle in radians (default: 0.0).
            weight (float): Scaling factor for the reward (default: 1.0).
            log (bool): Whether to log the deviation (default: False).

        Returns:
            torch.Tensor: Reward value for minimizing waist pitch deviation.
        """
        self.waist_pitch_index = self.dof_names.index('waist_pitch_joint')  # Rotates around Y-axis

        # Extract waist pitch joint angle
        waist_pitch_angle = self.dof_pos[:, self.waist_pitch_index]  # [num_envs]

        # Compute penalty for deviation from target pitch
        deviation_penalty = torch.abs(waist_pitch_angle - target_pitch)  # Absolute deviation

        # Log deviation for debugging
        if log:
            print(f"Waist Pitch Deviation: {torch.mean(deviation_penalty).item()}")

        # Compute final reward (exponential penalty with weight)
        reward = weight * torch.exp(-2.0 * deviation_penalty)
        return reward
        
    def _reward_minimize_waist_roll_deviation(self, target_roll=0.0, weight=1.0, log=False):
        """
        Penalize deviation of waist_pitch_joint from a target angle.

        Args:
            target_pitch (float): Desired waist pitch angle in radians (default: 0.0).
            weight (float): Scaling factor for the reward (default: 1.0).
            log (bool): Whether to log the deviation (default: False).

        Returns:
            torch.Tensor: Reward value for minimizing waist pitch deviation.
        """
        self.waist_roll_index = self.dof_names.index('waist_roll_joint')  # Rotates around Y-axis

        # Extract waist pitch joint angle
        waist_roll_angle = self.dof_pos[:, self.waist_roll_index]  # [num_envs]

        # Compute penalty for deviation from target pitch
        deviation_penalty = torch.abs(waist_roll_angle - target_roll)  # Absolute deviation

        # Log deviation for debugging
        if log:
            print(f"Waist Pitch Deviation: {torch.mean(deviation_penalty).item()}")

        # Compute final reward (exponential penalty with weight)
        reward = weight * torch.exp(-2.0 * deviation_penalty)

        return reward
    
    def _reward_minimize_waist_yaw_deviation(self, target_yaw=0.0, weight=1.0, log=False):
        """
        Penalize deviation of waist_pitch_joint from a target angle.

        Args:
            target_pitch (float): Desired waist pitch angle in radians (default: 0.0).
            weight (float): Scaling factor for the reward (default: 1.0).
            log (bool): Whether to log the deviation (default: False).

        Returns:
            torch.Tensor: Reward value for minimizing waist pitch deviation.
        """
        self.waist_yaw_index = self.dof_names.index('waist_yaw_joint')  # Rotates around Y-axis

        # Extract waist pitch joint angle
        waist_yaw_angle = self.dof_pos[:, self.waist_yaw_index]  # [num_envs]

        # Compute penalty for deviation from target pitch
        deviation_penalty = torch.abs(waist_yaw_angle - target_yaw)  # Absolute deviation

        # Log deviation for debugging
        if log:
            print(f"Waist Pitch Deviation: {torch.mean(deviation_penalty).item()}")

        # Compute final reward (exponential penalty with weight)
        reward = weight * torch.exp(-2.0 * deviation_penalty)

        return reward
    
    def _reward_torso_yaw_smoothness(self):
        """Discourages excessive or jerky torso yaw motion."""
        torso_yaw_idx = self.dof_names.index("waist_yaw_joint")
        yaw_vel = self.dof_vel[:, torso_yaw_idx]
        yaw_acc = self.dof_acc[:, torso_yaw_idx] if hasattr(self, "dof_acc") else torch.zeros_like(yaw_vel)
        
        smoothness_reward = -0.5 * yaw_vel**2 - 0.01 * yaw_acc**2
        return smoothness_reward
    
    def _extract_upper_body_angular_velocity(self):
        """
        Extract angular velocities (roll rate, pitch rate, yaw rate) for the pelvis,
        waist_roll_link, and torso_link individually.

        Returns:
            tuple: Summed absolute angular velocities and individual angular velocities
                for pelvis, waist, and torso.
        """

        # Extract indices for each body part separately
        pelvis_idx = self.body_names.index('pelvis')
        waist_idx = self.body_names.index('waist_roll_link')
        torso_idx = self.body_names.index('torso_link')

        # Extract angular velocity states for each body part
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]

        pelvis_state = self.rigid_body_states_view[:, pelvis_idx, :]
        # print(f"pelvis_state: {pelvis_state}")
        waist_state = self.rigid_body_states_view[:, waist_idx, :]
        torso_state = self.rigid_body_states_view[:, torso_idx, :]

        # Extract angular velocity components (indices 10:13)
        pelvis_ang_vel = pelvis_state[:, 10:13]  # Shape: [num_envs, 3]
        waist_ang_vel = waist_state[:, 10:13]    # Shape: [num_envs, 3]
        torso_ang_vel = torso_state[:, 10:13]    # Shape: [num_envs, 3]

        # Compute the sum of absolute angular velocities
        upper_body_ang_vel_sum = torch.abs(pelvis_ang_vel) + torch.abs(waist_ang_vel) + torch.abs(torso_ang_vel)

        return upper_body_ang_vel_sum, pelvis_ang_vel, waist_ang_vel, torso_ang_vel

    def _reward_orientation(self):
        # 假设 projected_gravity 已经是归一化向量在根坐标系下的 [gx, gy, gz]
        pitch_roll_err = torch.sum(self.projected_gravity[:, :2]**2, dim=1)   # (B,)
        pitch_roll_vel = torch.sum(self.base_ang_vel[:, :2]**2, dim=1)        # (B,)
        
        k_angle = 300.0   # 角度误差强度
        k_vel   = 100.0    # 角速度误差强度
        
        r_angle = torch.exp(-k_angle * pitch_roll_err)
        r_vel   = torch.exp(-k_vel   * pitch_roll_vel)
        
        # 取两项几何平均，避免任何一项为 0
        return torch.sqrt(r_angle * r_vel)

def flip_g1_actor_obs(obs):
    if obs is None:
        return obs

    flipped_obs = torch.zeros_like(obs)
    # base_ang_vel
    flipped_obs[..., :3] = obs[..., :3]
    flipped_obs[..., 0] *= -1  # Flip the x component of base_ang_vel
    flipped_obs[..., 2] *= -1  # Flip the z component of base_ang_vel
    # projected_gravity
    flipped_obs[..., 3:6] = obs[..., 3:6]
    flipped_obs[..., 4] *= -1  # Flip the y component of projected_gravity

    # command
    flipped_obs[..., 6:9] = obs[..., 6:9]
    flipped_obs[..., 7] *= -1  # Flip the vy component of command
    flipped_obs[..., 8] *= -1  # Flip the wz component of command

    # dof_pos
    # legs
    flipped_obs[..., 9:15] = obs[..., 15:21]
    flipped_obs[..., 15:21] = obs[..., 9:15]
    # waist
    flipped_obs[..., 21:24] = obs[..., 21:24]
    # arm
    flipped_obs[..., 24:27] = obs[..., 27:30]
    flipped_obs[..., 27:30] = obs[..., 24:27]

    # dof_vel
    # legs
    flipped_obs[..., 30:36] = obs[..., 36:42]
    flipped_obs[..., 36:42] = obs[..., 30:36]
    # waist
    flipped_obs[..., 42:45] = obs[..., 42:45]
    # arm
    flipped_obs[..., 45:48] = obs[..., 48:51]
    flipped_obs[..., 48:51] = obs[..., 45:48]

    # actions
    # legs
    flipped_obs[..., 51:57] = obs[..., 57:63]
    flipped_obs[..., 57:63] = obs[..., 51:57]
    # waist
    flipped_obs[..., 63:66] = obs[..., 63:66]
    # arm
    flipped_obs[..., 66:69] = obs[..., 69:72]
    flipped_obs[..., 69:72] = obs[..., 66:69]

    vel_offset = act_offset = 21
    base_offset = 9
    # Flip the sign of specific bases in the observation
    # hip_roll, hip_yaw, ankle_roll, waist_yaw, waist_roll, shoulder_roll
    for base in [1, 2, 5, 7, 8, 11, 12, 13, 16, 19]:
        flipped_obs[..., base + base_offset] *= -1
        flipped_obs[..., base + base_offset + vel_offset] *= -1
        flipped_obs[..., base + base_offset + vel_offset + act_offset] *= -1
    return torch.cat([obs, flipped_obs], dim=0)


def flip_g1_critic_obs(obs):
    if obs is None:
        return obs

    flipped_obs = torch.zeros_like(obs)
    # base_lin_vel
    flipped_obs[..., :3] = obs[..., :3]
    flipped_obs[..., 1] *= -1  # Flip the y component of base_lin_vel
    # base_ang_vel
    flipped_obs[..., 3:6] = obs[..., 3:6]
    flipped_obs[..., 3] *= -1  # Flip the x component of base_ang_vel
    flipped_obs[..., 5] *= -1  # Flip the z component of base_ang_vel
    # projected_gravity
    flipped_obs[..., 6:9] = obs[..., 6:9]
    flipped_obs[..., 7] *= -1  # Flip the y component of projected_gravity
    # command
    flipped_obs[..., 9:12] = obs[..., 9:12]
    flipped_obs[..., 10] *= -1  # Flip the vy component of command
    flipped_obs[..., 11] *= -1  # Flip the wz component of command

    # dof_pos
    # legs
    flipped_obs[..., 12:18] = obs[..., 18:24]
    flipped_obs[..., 18:24] = obs[..., 12:18]
    # waist
    flipped_obs[..., 24:27] = obs[..., 24:27]
    # arm
    flipped_obs[..., 27:30] = obs[..., 30:33]
    flipped_obs[..., 30:33] = obs[..., 27:30]

    # dof_vel
    # legs
    flipped_obs[..., 33:39] = obs[..., 39:45]
    flipped_obs[..., 39:45] = obs[..., 33:39]
    # waist
    flipped_obs[..., 45:48] = obs[..., 45:48]
    # arm
    flipped_obs[..., 48:51] = obs[..., 51:54]
    flipped_obs[..., 51:54] = obs[..., 48:51]

    # actions
    # legs
    flipped_obs[..., 54:60] = obs[..., 60:66]
    flipped_obs[..., 60:66] = obs[..., 54:60]
    # waist
    flipped_obs[..., 66:69] = obs[..., 66:69]
    # arm
    flipped_obs[..., 69:72] = obs[..., 72:75]
    flipped_obs[..., 72:75] = obs[..., 69:72]

    vel_offset = act_offset = 21
    # Flip the sign of specific bases in the observation
    # hip_roll, hip_yaw, ankle_roll, waist_yaw, waist_roll, shoulder_roll
    base_offset = 12
    for base in [1, 2, 5, 7, 8, 11, 12, 13, 16, 19]:
        flipped_obs[..., base + base_offset] *= -1
        flipped_obs[..., base + vel_offset + base_offset] *= -1
        flipped_obs[..., base_offset + vel_offset + act_offset] *= -1
    return torch.cat([obs, flipped_obs], dim=0)


def flip_g1_actions(actions):
    if actions is None:
        return None

    flip_actions = torch.zeros_like(actions)
    # legs
    flip_actions[..., :6] = actions[..., 6:12]
    flip_actions[..., 6:12] = actions[..., :6]
    # waist
    flip_actions[..., 12:15] = actions[..., 12:15]
    # arm
    flip_actions[..., 15:18] = actions[..., 18:21]
    flip_actions[..., 18:21] = actions[..., 15:18]

    # hip_roll, hip_yaw, ankle_roll, waist_yaw, waist_roll, shoulder_roll
    for base in [1, 2, 5, 7, 8, 11, 12, 13, 16, 19]:
        flip_actions[..., base] *= -1
    return torch.cat([actions, flip_actions], dim=0)
    
def data_augmentation_func_g1(obs, actions, env, obs_type):
        if obs_type == "policy":
            obs_batch = flip_g1_actor_obs(obs)
        else:
            obs_batch = flip_g1_critic_obs(obs)

        mean_actions_batch = flip_g1_actions(actions)
        return (obs_batch, mean_actions_batch)
