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
    # Constants for indexing into the motion data - specific to 36-value format
    # 3 + 4 + 12 + 3 + 3 + 12 = 37
    FRAMES_TO_LOAD = 36  # Number of frames to load from each file
    
    # Sizes of each component
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12  # 19 - 7 (root position and rotation)
    
    # Derived velocities - these will be computed
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE # 3

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE # 7

    JOINT_POS_START_IDX = ROOT_ROT_END_IDX
    JOINT_POS_END_IDX = JOINT_POS_START_IDX + JOINT_POS_SIZE # 19

    LINEAR_VEL_START_IDX = JOINT_POS_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE # 22 

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE # 25

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE # 37

    
    
    def __init__(
            self,
            device,
            time_between_frames,
            data_dir='',
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob('datasets/motion_files2/*'),
            ):
        """Expert dataset provides AMP observations from motion dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        
        # Values to store for each trajectory
        self.trajectories = []
        self.extended_traj = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            
            try:
                # Handle different file formats - assume text file with space/comma-separated values
                with open(motion_file, "r") as f:
                    # Try to detect if this is JSON first
                    try:
                        motion_json = json.load(f)
                        motion_data = np.array(motion_json["Frames"])
                        frame_duration = float(motion_json.get("FrameDuration", 1.0/30.0))  # Default to 30fps
                        motion_weight = float(motion_json.get("MotionWeight", 1.0))
                    except json.JSONDecodeError:
                        # If not JSON, check file extension
                        if motion_file.endswith('.csv'):
                            # Reset file pointer and read as CSV
                            f.seek(0)
                            motion_data = []
                            for line in f:


                                # Split by comma and convert to float
                                values = [float(x) for x in line.strip().split(',')]
                                # if len(values) == 36:  # Ensure line has expected number of values
                                motion_data.append(values)
                        else:
                            # Assume it's a text file with space-separated values
                            f.seek(0)
                            lines = f.readlines()
                            motion_data = []
                            for line in lines:
                                # Clean the line and split by whitespace
                                values = [float(x) for x in line.strip().split()]
                                # if len(values) == 36:  # Ensure line has expected number of values
                                motion_data.append(values)
                        motion_data = np.array(motion_data)
                        frame_duration = 1.0/30.0  # Assume 30fps for text files
                        motion_weight = 1.0
                
                # Normalize and standardize quaternions
                for f_i in range(motion_data.shape[0]):
                    root_rot = self.get_root_rot(motion_data[f_i])
                    root_rot = pose3d.QuaternionNormalize(root_rot)
                    root_rot = motion_util.standardize_quaternion(root_rot)
                    motion_data[
                        f_i,
                        self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot
                
                # Compute velocities from position differences
                frame_rate = 30.0  # 30Hz
                dt = 1.0 / frame_rate
                
                # Only compute velocities if we have at least 2 frames
                if motion_data.shape[0] > 1:
                    # Compute linear velocities (root position)
                    lin_vel = np.zeros((motion_data.shape[0], self.LINEAR_VEL_SIZE))
                    # Skip first frame for velocities since we need two frames to compute
                    for f_i in range(1, motion_data.shape[0]):
                        pos_curr = self.get_root_pos(motion_data[f_i])
                        pos_prev = self.get_root_pos(motion_data[f_i-1])
                        lin_vel[f_i] = (pos_curr - pos_prev) / dt
                    
                    # Compute angular velocities (from quaternions)
                    ang_vel = np.zeros((motion_data.shape[0], self.ANGULAR_VEL_SIZE))
                    for f_i in range(1, motion_data.shape[0]):
                        quat_curr = self.get_root_rot(motion_data[f_i])
                        quat_prev = self.get_root_rot(motion_data[f_i-1])
                        
                        # Get the relative rotation between frames
                        quat_diff = transformations.quaternion_multiply(
                            quat_curr,
                            transformations.quaternion_inverse(quat_prev)
                        )
                        
                        # Convert to axis-angle representation
                        axis, angle = pose3d.QuaternionToAxisAngle(quat_diff)
                        
                        # Angular velocity is axis * angle / dt
                        ang_vel[f_i] = axis * angle / dt
                    
                    # Compute joint velocities
                    joint_vel = np.zeros((motion_data.shape[0], self.JOINT_VEL_SIZE))
                    for f_i in range(1, motion_data.shape[0]):
                        joint_curr = self.get_joint_pose(motion_data[f_i])
                        joint_prev = self.get_joint_pose(motion_data[f_i-1])
                        joint_vel[f_i] = (joint_curr - joint_prev) / dt
                    
                    # First frame velocities are the same as second frame to avoid zeros
                    lin_vel[0] = lin_vel[1]
                    ang_vel[0] = ang_vel[1]
                    joint_vel[0] = joint_vel[1]
                else:
                    # If only one frame, all velocities are zero
                    lin_vel = np.zeros((motion_data.shape[0], self.LINEAR_VEL_SIZE))
                    ang_vel = np.zeros((motion_data.shape[0], self.ANGULAR_VEL_SIZE))
                    joint_vel = np.zeros((motion_data.shape[0], self.JOINT_VEL_SIZE))
                
                # Store all computed velocities as properties in the class
                self.lin_vel = torch.tensor(lin_vel, dtype=torch.float32, device=device)
                self.ang_vel = torch.tensor(ang_vel, dtype=torch.float32, device=device)
                self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)
                
                # Store trajectory data (without the first 7 dimensions for regular traj)
                self.trajectories.append(torch.tensor(
                    motion_data[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX],
                    dtype=torch.float32, device=device))
                
                # Store full trajectory data with velocities
                # Create an extended motion data array that includes original data and computed velocities
                extended_motion_data = np.zeros((motion_data.shape[0], motion_data.shape[1] + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE))
                extended_motion_data[:, :motion_data.shape[1]] = motion_data  # Original data

                # Add velocities
                extended_motion_data[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX] = lin_vel
                extended_motion_data[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX] = ang_vel
                extended_motion_data[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX] = joint_vel
                
                self.extended_traj.append(torch.tensor(
                    extended_motion_data[:, self.JOINT_POS_START_IDX:self.JOINT_VEL_END_IDX],
                    dtype=torch.float32, device=device))

                # Store extended data as tensor
                self.trajectories_full.append(torch.tensor(
                    extended_motion_data,
                    dtype=torch.float32, device=device))
                
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(motion_weight)
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

                print(f"Loaded {traj_len}s motion from {motion_file}.")
                
            except Exception as e:
                print(f"Error loading {motion_file}: {e}")
                continue
        
        # Handle empty trajectory case
        if not self.trajectory_weights:
            raise ValueError("No valid motion files were loaded")
            
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
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

        self.all_trajectories_full = torch.vstack(self.trajectories_full) if self.trajectories_full else torch.tensor([])
        print(f'trajectories shape: {self.trajectories[0].shape}')
        print(f'trajectories_full shape: {self.trajectories_full[0].shape}')
        print(f'all_trajectories_full shape: {self.all_trajectories_full.shape}')
        
    @staticmethod
    def get_root_pos(pose):
        """Get root position from a pose vector."""
        return pose[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_pos_batch(poses):
        """Get root positions from a batch of pose vectors."""
        return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot(pose):
        """Get root rotation from a pose vector."""
        return pose[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_root_rot_batch(poses):
        """Get root rotations from a batch of pose vectors."""
        return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_joint_pose(pose):
        """Get joint poses from a pose vector."""
        return pose[AMPLoader.JOINT_POS_START_IDX:AMPLoader.JOINT_POS_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        """Get joint poses from a batch of pose vectors."""
        return poses[:, AMPLoader.JOINT_POS_START_IDX:AMPLoader.JOINT_POS_END_IDX]
    
    @staticmethod
    def get_linear_vel(pose):
        return pose[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]
    
    @staticmethod
    def get_linear_vel_batch(pose):
        return pose[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]
    
    @staticmethod
    def get_angular_vel(pose):
        return pose[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  

    @staticmethod
    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]
    
    @staticmethod
    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

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

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

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
        
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        idx_high = min(idx_high, n - 1)  # Ensure we don't go out of bounds
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """Returns full frames for the given trajectories at the specified times."""
        """找到时间点对应的各项观测量，并在低位高位之间进行插值，返回插值后观测量"""
        p = times / np.maximum(self.trajectory_lens[traj_idxs], 1e-10)  # Avoid division by zero
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        
        # Clamp indices to valid range
        idx_low = np.clip(idx_low, 0, n - 1)  
        idx_high = np.clip(idx_high, 0, n - 1)
        
        # Initialize tensors to hold the interpolation values
        # For positions and orientations
        all_frame_pos_starts = torch.zeros(len(traj_idxs), self.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), self.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), self.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), self.ROT_SIZE, device=self.device)
        all_frame_joint_starts = torch.zeros(len(traj_idxs), self.JOINT_POS_SIZE, device=self.device)
        all_frame_joint_ends = torch.zeros(len(traj_idxs), self.JOINT_POS_SIZE, device=self.device)
        
        # For velocities
        all_frame_lin_vel_starts = torch.zeros(len(traj_idxs), self.LINEAR_VEL_SIZE, device=self.device)
        all_frame_lin_vel_ends = torch.zeros(len(traj_idxs), self.LINEAR_VEL_SIZE, device=self.device)
        all_frame_ang_vel_starts = torch.zeros(len(traj_idxs), self.ANGULAR_VEL_SIZE, device=self.device)
        all_frame_ang_vel_ends = torch.zeros(len(traj_idxs), self.ANGULAR_VEL_SIZE, device=self.device)
        all_frame_joint_vel_starts = torch.zeros(len(traj_idxs), self.JOINT_VEL_SIZE, device=self.device)
        all_frame_joint_vel_ends = torch.zeros(len(traj_idxs), self.JOINT_VEL_SIZE, device=self.device)
        
        
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            
            # Extract components for each trajectory's frames - positions and orientations
            all_frame_pos_starts[traj_mask] = self.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = self.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            
            all_frame_rot_starts[traj_mask] = self.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = self.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            
            all_frame_joint_starts[traj_mask] = self.get_joint_pose_batch(trajectory[idx_low[traj_mask]])
            all_frame_joint_ends[traj_mask] = self.get_joint_pose_batch(trajectory[idx_high[traj_mask]])
            
            # Extract velocity components
            all_frame_lin_vel_starts[traj_mask] = self.get_linear_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_lin_vel_ends[traj_mask] = self.get_linear_vel_batch(trajectory[idx_high[traj_mask]])
            
            all_frame_ang_vel_starts[traj_mask] = self.get_angular_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_ang_vel_ends[traj_mask] = self.get_angular_vel_batch(trajectory[idx_high[traj_mask]])
            
            all_frame_joint_vel_starts[traj_mask] = self.get_joint_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_joint_vel_ends[traj_mask] = self.get_joint_vel_batch(trajectory[idx_high[traj_mask]])
        
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        # Interpolate position linearly
        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        
        # Use quaternion interpolation for rotation
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        
        # Interpolate joint positions and velocities linearly
        joint_blend = self.slerp(all_frame_joint_starts, all_frame_joint_ends, blend)
        lin_vel_blend = self.slerp(all_frame_lin_vel_starts, all_frame_lin_vel_ends, blend)
        ang_vel_blend = self.slerp(all_frame_ang_vel_starts, all_frame_ang_vel_ends, blend)
        joint_vel_blend = self.slerp(all_frame_joint_vel_starts, all_frame_joint_vel_ends, blend)
        
        # Combine all components
        return torch.cat([
            pos_blend, 
            rot_blend, 
            joint_blend, 
            lin_vel_blend, 
            ang_vel_blend, 
            joint_vel_blend
        ], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        """Returns a batch of random full frames."""
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

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
                s = self.preloaded_s[idxs, self.JOINT_POS_START_IDX:self.JOINT_VEL_END_IDX]
                
                # Add root height (Z coordinate)
                s = torch.cat([
                    s,
                    self.preloaded_s[idxs, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]], dim=-1)
                
                # Same for next state
                s_next = self.preloaded_s_next[idxs, self.JOINT_POS_START_IDX:self.JOINT_VEL_END_IDX]
                s_next = torch.cat([
                    s_next,
                    self.preloaded_s_next[idxs, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]], dim=-1)
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                
                for traj_idx, frame_time in zip(traj_idxs, times):
                    frame = self.get_frame_at_time(traj_idx, frame_time)
                    next_frame = self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames)
                    
                    # We need to get the root height for each frame
                    full_frame = self.get_full_frame_at_time(traj_idx, frame_time)
                    full_next_frame = self.get_full_frame_at_time(traj_idx, frame_time + self.time_between_frames)
                    
                    # Append joint positions and root height
                    s.append(torch.cat([frame, full_frame[self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]]))
                    s_next.append(torch.cat([next_frame, full_next_frame[self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]]))
                
                s = torch.stack(s)
                s_next = torch.stack(s_next)
            
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.extended_traj[0].shape[1] + 1 # Joint positions + root height