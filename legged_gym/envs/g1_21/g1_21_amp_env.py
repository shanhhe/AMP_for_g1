
from legged_gym.envs.base.g1_legged_robot import G1LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G121AMPRobot(G1LeggedRobot):
    
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
        k_vel   = 1.0    # 角速度误差强度
        
        r_angle = torch.exp(-k_angle * pitch_roll_err)
        r_vel   = torch.exp(-k_vel   * pitch_roll_vel)
        
        # 取两项几何平均，避免任何一项为 0
        return torch.sqrt(r_angle * r_vel)