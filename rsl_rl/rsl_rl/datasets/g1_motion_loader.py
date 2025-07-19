import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util


class AMPLoader:

    def set_data_index(self):
        """Set data index for the motion data."""
        # Constants for indexing into the motion data - specific to 36-value format
        # 3 + 4 + num_dofs + 3 + 3 + num_dofs = 55
        if self.datatype == "Joint":
            self.JOINT_POS_SIZE = 21
        else:
            self.JOINT_POS_SIZE = 14 * 4 # 10 joints, each with 3 values (x, y, z)

        self.JOINT_VEL_SIZE = self.JOINT_POS_SIZE

        # Sizes of each component
        self.POS_SIZE = 3
        self.ROT_SIZE = 4
    
        # Derived velocities - these will be computed
        self.LINEAR_VEL_SIZE = 3
        self.ANGULAR_VEL_SIZE = 3

        self.ROOT_POS_START_IDX = 0
        self.ROOT_POS_END_IDX = self.ROOT_POS_START_IDX + self.POS_SIZE # 3

        self.ROOT_ROT_START_IDX = self.ROOT_POS_END_IDX
        self.ROOT_ROT_END_IDX = self.ROOT_ROT_START_IDX + self.ROT_SIZE # 7

        self.JOINT_POS_START_IDX = self.ROOT_ROT_END_IDX
        self.JOINT_POS_END_IDX = self.JOINT_POS_START_IDX + self.JOINT_POS_SIZE # 7 + num_dofs

        self.LINEAR_VEL_START_IDX = self.JOINT_POS_END_IDX
        self.LINEAR_VEL_END_IDX = self.LINEAR_VEL_START_IDX + self.LINEAR_VEL_SIZE # 10 + num_dofs

        self.ANGULAR_VEL_START_IDX = self.LINEAR_VEL_END_IDX
        self.ANGULAR_VEL_END_IDX = self.ANGULAR_VEL_START_IDX + self.ANGULAR_VEL_SIZE # 13 + num_dofs

        self.JOINT_VEL_START_IDX = self.ANGULAR_VEL_END_IDX
        self.JOINT_VEL_END_IDX = self.JOINT_VEL_START_IDX + self.JOINT_VEL_SIZE # 13 + 2 * num_dofs

    
    def __init__(
            self,
            device,
            time_between_frames,
            data_dir='',
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob('datasets/motion_files2/*'),
            selected_joint_indices=None,  # 新增参数：你想要保留的关节索引列表
            datatype='Joint',
            ):
        """Expert dataset provides AMP observations from motion dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.selected_joint_indices = selected_joint_indices
        self.num_dofs = len(selected_joint_indices)
        self.datatype = datatype
        self.set_data_index()
        
        # Values to store for each trajectory
        self.trajectories = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            
            # Handle different file formats - assume text file with space/comma-separated values
            with open(motion_file, "r") as f:
                if motion_file.endswith('.csv'):
                    # Reset file pointer and read as CSV
                    f.seek(0)
                    motion_data = []
                    for line in f:
                        # Split by comma and convert to float
                        values = [float(x) for x in line.strip().split(',')]
                        motion_data.append(values)
                        # print(f"Loaded {len(values)} values from {motion_file}.")
                motion_data = np.array(motion_data)
                frame_duration = 0.02  # Assume 30fps for text files
                motion_weight = 1.0

            self.get_data(motion_data, device, frame_duration, motion_weight, motion_file, i)

        # Handle empty trajectory case
        if not self.trajectory_weights:
            raise ValueError("No valid motion files were loaded")
        
        # 以时间长度为权重进行归一化
        for i in range(len(self.trajectory_weights)):
            if self.trajectory_lens[i] <= 0:
                raise ValueError(f"Trajectory {self.trajectory_names[i]} has non-positive length: {self.trajectory_lens[i]}")
            self.trajectory_weights[i] *= self.trajectory_lens[i]
            
        # Trajectory weights are used to sample some trajectories more than others
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions
        self.preload_transitions = preload_transitions # True
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

        print(f'trajectories shape: {self.trajectories[0].shape}, datatype: {self.datatype}')    

    def get_data(self, motion_data, device, frame_duration, motion_weight, motion_file, i):
        # Normalize and standardize quaternions
        for f_i in range(motion_data.shape[0]):
            root_rot = self.get_root_rot(motion_data[f_i])
            root_rot = pose3d.QuaternionNormalize(root_rot)
            root_rot = motion_util.standardize_quaternion(root_rot)
            motion_data[
                f_i,
                self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot
        
        # Compute velocities from position differences
        frame_rate = 50  # 50Hz
        dt = 1.0 / frame_rate
    

        self.trajectories.append(torch.tensor(
            motion_data,
            dtype=torch.float32, device=device))
        
        self.trajectory_idxs.append(i)
        self.trajectory_weights.append(motion_weight)
        self.trajectory_frame_durations.append(frame_duration)
        traj_len = (motion_data.shape[0] - 1) * frame_duration
        self.trajectory_lens.append(traj_len)
        self.trajectory_num_frames.append(float(motion_data.shape[0]))

        print(f"Loaded {traj_len}s motion from {motion_file}.")


    def get_root_pos(self, pose):
        """Get root position from a pose vector."""
        return pose[self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_pos_batch(self, poses):
        """Get root positions from a batch of pose vectors."""
        return poses[:, self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_rot(self, pose):
        """Get root rotation from a pose vector."""
        return pose[self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_root_rot_batch(self, poses):
        """Get root rotations from a batch of pose vectors."""
        return poses[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_joint_pose(self, pose):
        """Get joint poses from a pose vector."""
        return pose[self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]

    def get_joint_pose_batch(self, poses):
        """Get joint poses from a batch of pose vectors."""
        return poses[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]
    
    def get_linear_vel(self, pose):
        return pose[self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]
    
    def get_linear_vel_batch(self, pose):
        return pose[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]
    
    def get_angular_vel(self, pose):
        return pose[self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(self, poses):
        return poses[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]
    
    def get_joint_vel(self, pose):
        return pose[self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(self, poses):
        return poses[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        """Linear interpolation between values."""
        return (1.0 - blend) * val0 + blend * val1

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        idx_high = min(idx_high, n - 1)  # Ensure we don't go out of bounds
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / np.maximum(self.trajectory_lens[traj_idxs], 1e-10)  # Avoid division by zero
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        
        # Clamp indices to valid range
        idx_low = np.clip(idx_low, 0, n - 1)
        idx_high = np.clip(idx_high, 0, n - 1)
        all_frame_starts = torch.zeros(len(traj_idxs), self.trajectories[0].shape[1], device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.trajectories[0].shape[1], device=self.device)
        

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)
   
    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)
    
    def get_full_frame_batch(self, num_frames):
        """Returns a batch of random full frames."""
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """
        # Get original frame data
        root_pos0, root_pos1 = self.get_root_pos(frame0), self.get_root_pos(frame1)
        root_rot0, root_rot1 = self.get_root_rot(frame0), self.get_root_rot(frame1)
        joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
        
        # Get velocity data - offset by original data length
        offset = self.ROOT_POS_END_IDX + self.ROT_SIZE + self.JOINT_POS_SIZE
        
        # Extract velocities using tensor slicing
        lin_vel0 = frame0[offset:offset+self.LINEAR_VEL_SIZE]
        lin_vel1 = frame1[offset:offset+self.LINEAR_VEL_SIZE]
        
        offset += self.LINEAR_VEL_SIZE
        ang_vel0 = frame0[offset:offset+self.ANGULAR_VEL_SIZE]
        ang_vel1 = frame1[offset:offset+self.ANGULAR_VEL_SIZE]
        
        offset += self.ANGULAR_VEL_SIZE
        joint_vel0 = frame0[offset:offset+self.JOINT_VEL_SIZE]
        joint_vel1 = frame1[offset:offset+self.JOINT_VEL_SIZE]

        # Blend positions linearly
        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        
        # Use quaternion interpolation for rotation
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
            
        # Blend joint positions and velocities linearly
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_lin_vel = self.slerp(lin_vel0, lin_vel1, blend)
        blend_ang_vel = self.slerp(ang_vel0, ang_vel1, blend)
        blend_joint_vel = self.slerp(joint_vel0, joint_vel1, blend)

        # Concatenate all blended components
        return torch.cat([
            blend_root_pos, 
            blend_root_rot, 
            blend_joints, 
            blend_lin_vel, 
            blend_ang_vel, 
            blend_joint_vel
        ])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                
                # Get only the joint positions for the state
                # if self.datatype == "Joint":
                s = self.preloaded_s[idxs, self.JOINT_POS_START_IDX:]
                # s = self.preloaded_s[idxs, self.JOINT_POS_START_IDX:self.ANGULAR_VEL_END_IDX]

                # else:
                #     s = self.preloaded_s[idxs, self.JOINT_POS_START_IDX:self.ANGULAR_VEL_END_IDX]
                
                # Add root height (Z coordinate)
                s = torch.cat([
                    s,
                    self.preloaded_s[idxs, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]], dim=-1)
                
                # Same for next state
                # if self.datatype == "Joint":
                s_next = self.preloaded_s_next[idxs, self.JOINT_POS_START_IDX:]
                # s_next = self.preloaded_s_next[idxs, self.JOINT_POS_START_IDX:self.ANGULAR_VEL_END_IDX]
                # else:
                #     s_next = self.preloaded_s_next[idxs, self.JOINT_POS_START_IDX:self.ANGULAR_VEL_END_IDX]
                s_next = torch.cat([
                    s_next,
                    self.preloaded_s_next[idxs, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]], dim=-1)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        # if self.datatype == "Joint":
        return self.trajectories[0].shape[1] - 6
        # return self.trajectories[0].shape[1] - 6 - 11 * 4 # Joint positions + lin_vel + ang_vel+ root height
        # else:
        #     return self.trajectories[0].shape[1] - 6 - self.JOINT_POS_SIZE