
from legged_gym.envs.base.g1leg_legged_robot import G1LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1LEGAMPRobot(G1LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        '''_init_foot obtains the current state of the robot's rigid bodies,
          reshapes it, and extracts information (positions and velocities) 
          for the feet using the provided feet_indices.
        '''
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        # Reward for contact
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    
    def _reward_straight_stance_phase(self):
        """
        Reward function that encourages a straight knee in the support leg during stance phase
        while the other leg is in swing phase.
        """
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Identify which leg is in stance and which is in swing based on phase information
        left_leg_stance = self.leg_phase[:, 0] < 0.55  # Left leg in stance phase
        right_leg_stance = self.leg_phase[:, 1] < 0.55  # Right leg in stance phase
        
        # Check if contact matches the expected phase
        left_leg_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_leg_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0
        
        # Get knee joint angles
        left_knee_angle = torch.abs(self.dof_pos[:, self.knee_indices[0]])
        right_knee_angle = torch.abs(self.dof_pos[:, self.knee_indices[1]])
        
        # Calculate straightness - closer to zero means straighter leg
        # Exponential reward that peaks when the knee is straight (angle close to 0)
        left_leg_straightness = torch.exp(-5.0 * left_knee_angle)
        right_leg_straightness = torch.exp(-5.0 * right_knee_angle)
        
        # Apply reward only when the leg is in stance phase and has ground contact
        # while the other leg is swinging (not in contact)
        left_stance_condition = left_leg_stance & left_leg_contact & ~right_leg_contact
        right_stance_condition = right_leg_stance & right_leg_contact & ~left_leg_contact
        
        # Apply the reward when conditions are met
        reward += left_stance_condition * left_leg_straightness
        reward += right_stance_condition * right_leg_straightness
        
        return reward
    
    def _reward_swing_height(self):
        """Reward appropriate swing height for the non-stance foot."""
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Identify swing legs
        left_leg_swing = self.leg_phase[:, 0] >= 0.55
        right_leg_swing = self.leg_phase[:, 1] >= 0.55
        
        # Get foot heights
        left_foot_height = self.feet_pos[:, 0, 2]  # Z-coordinate of left foot
        right_foot_height = self.feet_pos[:, 1, 2]  # Z-coordinate of right foot
        
        # Target height during mid-swing (about 10-15cm clearance)
        target_height = 0.12
        
        # Calculate height error
        left_height_error = torch.abs(left_foot_height - target_height)
        right_height_error = torch.abs(right_foot_height - target_height)
        
        # Apply reward based on how close the foot height is to target during swing
        reward += left_leg_swing * torch.exp(-10.0 * left_height_error)
        reward += right_leg_swing * torch.exp(-10.0 * right_height_error)
        
        return reward

    def _reward_stance_swing_coordination(self):
        """Reward proper alternating stance-swing pattern."""
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Check if one leg is in stance while other is in swing
        proper_coordination = (self.leg_phase[:, 0] < 0.55) ^ (self.leg_phase[:, 1] < 0.55)
        
        # Apply reward for proper coordination
        reward += proper_coordination.float()
        
        return reward
    
    def _reward_penalty_knee_hyperextension(self):
        """Penalty for knee hyperextension (going beyond straight)."""
        penalty = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Define threshold for hyperextension (slightly beyond straight)
        # This depends on your joint angle convention
        hyperextension_threshold = -0.05  # Negative if extension is negative in your setup
        
        # Check for hyperextension
        left_knee_hyperext = torch.min(self.dof_pos[:, self.knee_indices[0]] - hyperextension_threshold, torch.zeros_like(self.dof_pos[:, self.knee_indices[0]]))
        right_knee_hyperext = torch.min(self.dof_pos[:, self.knee_indices[1]] - hyperextension_threshold, torch.zeros_like(self.dof_pos[:, self.knee_indices[1]]))
        
        # Apply penalties
        penalty += torch.abs(left_knee_hyperext) + torch.abs(right_knee_hyperext)
        
        return -penalty

    def reward_cop_progression_single_link(self):
        """
        Reward function based on center of pressure progression for a single-link foot.
        """
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Get phase information
        left_phase = self.leg_phase[:, 0]
        right_phase = self.leg_phase[:, 1]
        
        # Calculate center of pressure for each foot
        # This requires some approximation based on contact forces and positions
        
        # Assuming we have access to contact positions and forces
        left_contact_forces = self.contact_forces[:, self.left_foot_indices, 2]
        right_contact_forces = self.contact_forces[:, self.right_foot_indices, 2]
        
        left_contact_positions = self.contact_positions[:, self.left_foot_indices]
        right_contact_positions = self.contact_positions[:, self.right_foot_indices]
        
        # Convert to foot-local coordinates
        left_foot_pos = self.feet_pos[:, 0].unsqueeze(1)
        right_foot_pos = self.feet_pos[:, 1].unsqueeze(1)
        
        left_local_contacts = left_contact_positions - left_foot_pos
        right_local_contacts = right_contact_positions - right_foot_pos
        
        # Calculate CoP as weighted average of contact positions
        left_total_force = torch.sum(left_contact_forces, dim=1)
        right_total_force = torch.sum(right_contact_forces, dim=1)
        
        # Avoid division by zero
        left_contact_mask = left_total_force > 1.0
        right_contact_mask = right_total_force > 1.0
        
        # Initialize CoP with zeros
        left_cop_x = torch.zeros(self.num_envs, device=self.device)
        right_cop_x = torch.zeros(self.num_envs, device=self.device)
        
        # Calculate CoP x-position (along foot length)
        for i in range(self.num_envs):
            if left_contact_mask[i]:
                weighted_sum = torch.sum(left_local_contacts[i, :, 0] * left_contact_forces[i])
                left_cop_x[i] = weighted_sum / left_total_force[i]
            
            if right_contact_mask[i]:
                weighted_sum = torch.sum(right_local_contacts[i, :, 0] * right_contact_forces[i])
                right_cop_x[i] = weighted_sum / right_total_force[i]
        
        # Define target CoP progression based on stance phase
        # We want CoP to move from heel (-0.08) to toe (0.08) during stance
        
        left_target_cop = torch.zeros_like(left_phase)
        right_target_cop = torch.zeros_like(right_phase)
        
        # Only apply during stance phase (0.0 to 0.6)
        left_stance_mask = (left_phase < 0.6) & left_contact_mask
        right_stance_mask = (right_phase < 0.6) & right_contact_mask
        
        # Normalized phase within stance (0.0 to 1.0)
        left_stance_norm = torch.zeros_like(left_phase)
        left_stance_norm[left_stance_mask] = left_phase[left_stance_mask] / 0.6
        
        right_stance_norm = torch.zeros_like(right_phase)
        right_stance_norm[right_stance_mask] = right_phase[right_stance_mask] / 0.6
        
        # Linear progression from heel to toe
        foot_length = 0.16  # Total length from heel to toe
        heel_pos = -0.08
        toe_pos = 0.08
        
        left_target_cop[left_stance_mask] = heel_pos + left_stance_norm[left_stance_mask] * (toe_pos - heel_pos)
        right_target_cop[right_stance_mask] = heel_pos + right_stance_norm[right_stance_mask] * (toe_pos - heel_pos)
        
        # Calculate error
        left_cop_error = torch.abs(left_cop_x - left_target_cop)
        right_cop_error = torch.abs(right_cop_x - right_target_cop)
        
        # Reward being close to target CoP
        left_cop_reward = torch.exp(-15.0 * left_cop_error) * left_stance_mask.float()
        right_cop_reward = torch.exp(-15.0 * right_cop_error) * right_stance_mask.float()
        
        # Combine rewards
        reward += left_cop_reward + right_cop_reward
        
        return reward