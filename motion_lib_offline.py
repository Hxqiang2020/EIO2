from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import os
import joblib
import glob

import motion.smpl_sim.poselib.core.rotation3d as pRot
import scipy.ndimage as filters
import torch

class MathEngine:

    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(q, axis=-1, keepdims=True)
        return q / (norms + 1e-12)
    
    @staticmethod
    def _compute_angular_velocity(rot, time_delta: float, guassian_filter=True):
        assert len(rot.shape) == 3
        r = torch.from_numpy(rot[:, :, [1, 2, 3, 0]]).float() # wxyz --> xyzw
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[:-1, :, :] = pRot.quat_mul_norm(r[1:, :, :], pRot.quat_inverse(r[:-1, :, :]))
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
        return angular_velocity.numpy()
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        assert len(p.shape) == 3
        velocity = np.gradient(p, axis=-3) / time_delta
        if guassian_filter:
            velocity = filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")
        
        return velocity
    
    @staticmethod
    def _compute_dof_vel(joint_pos: np.ndarray, time_delta: float) -> np.ndarray:
        assert len(joint_pos.shape) == 2
        dof_vel = (joint_pos[1:, :] - joint_pos[:-1, :]) / time_delta
        dof_vel = np.concatenate([dof_vel, dof_vel[-1:]], axis=0)
        return dof_vel
    
class MotionLibCfg:
    fps = 50
    motion_dir = "Datasets/target_data/randomization"

class MotionLibOffline:

    def __init__(self, motion_lib_cfg: MotionLibCfg = None):
        self.motion_lib_cfg = motion_lib_cfg


    def randomization_motion(self):

        file_list = glob.glob(os.path.join(self.motion_lib_cfg.motion_dir, "**", "*.pkl"), recursive=True)

        default_motion_fps = float(self.motion_lib_cfg.fps) # 50hz

        default_motion_dt = 1. / default_motion_fps # 1.0 / 50 = 0.02

        for curr_file in file_list:

            key = curr_file.split("/")[-1].rsplit(".pkl", 1)[0]

            folder = curr_file.split("/")

            assert ("randomization" in folder) and (folder[-1][-4:] == ".pkl")

            folder = "/".join(folder[folder.index("randomization") + 1: -1])

            tgt_folder = folder

            curr_motion = joblib.load(curr_file)

            body_pos = curr_motion.pop("body_pos")
            body_rot = curr_motion.pop("body_rot")
            body_ang_vel = curr_motion.pop("body_ang_vel")
            body_lin_vel = curr_motion.pop("body_lin_vel")
            reset_joint_pos = curr_motion.pop("reset_joint_pos")
            reset_joint_vel = curr_motion.pop("reset_joint_vel")
            reset_root_trans = curr_motion.pop("reset_root_trans")
            reset_root_rot = curr_motion.pop("reset_root_rot")

            curr_num_frames = body_pos.shape[0]

            default_motion_len_sec = (curr_num_frames - 1) * default_motion_dt

            sample_time_internals = np.arange(-1, curr_num_frames - 1) * default_motion_dt
            sample_time_internals = sample_time_internals + (np.random.rand(*sample_time_internals.shape).astype(sample_time_internals.dtype) * 2.0 - 1.0) * 0.003

            ##### 计算离采样时间步前后最近的数据帧索引，并根据时间差计算插帧权重

            frame_idx0, frame_idx1, blend = self.calc_frame_blend(
                sample_times_sec=sample_time_internals,
                motion_lengths_sec=default_motion_len_sec,
                num_frames=curr_num_frames,
                dt_sec=default_motion_dt,
            )   

            global_idx0 = frame_idx0 ##采样时间步前一帧索引
            global_idx1 = frame_idx1 ##采样时间步后一帧索引

            weight_body = blend[:, None, None]  # (N,1,1)
            weight_vec = blend[:, None]         # (N,1)

            ## 采样时间步前后帧数据
            body_pos0, body_pos1 = body_pos[global_idx0], body_pos[global_idx1]
            body_rot0, body_rot1 = body_rot[global_idx0], body_rot[global_idx1]
            body_lin_vel0, body_lin_vel1 = body_lin_vel[global_idx0], body_lin_vel[global_idx1]
            body_ang_vel0, body_ang_vel1 = body_ang_vel[global_idx0], body_ang_vel[global_idx1]
            reset_joint_pos0, reset_joint_pos1 = reset_joint_pos[global_idx0], reset_joint_pos[global_idx1]
            reset_joint_vel0, reset_joint_vel1 = reset_joint_vel[global_idx0], reset_joint_vel[global_idx1]
            reset_root_trans0, reset_root_trans1 = reset_root_trans[global_idx0], reset_root_trans[global_idx1]
            reset_root_rot0, reset_root_rot1 = reset_root_rot[global_idx0], reset_root_rot[global_idx1]
                        
            ## 利用采样时间步前后帧数据通过采样权重进行插帧
            curr_motion["body_pos"] = self.lerp(body_pos0, body_pos1, weight_body)
            curr_motion["body_rot"] = self.quat_slerp(body_rot0, body_rot1, weight_body)
            curr_motion["body_lin_vel"] = self.lerp(body_lin_vel0, body_lin_vel1, weight_body)
            curr_motion["body_ang_vel"] = self.lerp(body_ang_vel0, body_ang_vel1, weight_body)
            curr_motion["reset_joint_pos"] = self.lerp(reset_joint_pos0, reset_joint_pos1, weight_vec)
            curr_motion["reset_joint_vel"] = self.lerp(reset_joint_vel0, reset_joint_vel1, weight_vec)
            curr_motion["reset_root_trans"] = self.lerp(reset_root_trans0, reset_root_trans1, weight_vec)
            curr_motion["reset_root_rot"] = self.quat_slerp(reset_root_rot0, reset_root_rot1, weight_vec)

            os.makedirs(f"Datasets/target_data/randomization_fps/{tgt_folder}/", exist_ok=True)
            joblib.dump(curr_motion, f"Datasets/target_data/randomization_fps/{tgt_folder}/{key}.pkl")

    def calc_frame_blend(self, sample_times_sec, motion_lengths_sec, num_frames, dt_sec):

        sample_times_sec = np.array(sample_times_sec, copy=True)
        phase = sample_times_sec / motion_lengths_sec
        phase = np.clip(phase, 0.0, 1.0)
        sample_times_sec[sample_times_sec < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).astype(np.int64)
        frame_idx1 = np.minimum(frame_idx0 + 1, num_frames - 1)
        blend = np.clip((sample_times_sec - frame_idx0 * dt_sec) / dt_sec, 0.0, 1.0)
        
        return frame_idx0, frame_idx1, blend
    
    def lerp(self, value0: np.ndarray, value1: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return (np.float32(1.0) - weight) * value0 + weight * value1
    
    def quat_slerp(self, q0: np.ndarray, q1: np.ndarray, weight: np.ndarray, eps: float = 1e-8) -> np.ndarray:

        q0 = np.asarray(q0, dtype=np.float32)
        q1 = np.asarray(q1, dtype=np.float32)
        weight  = np.asarray(weight,  dtype=np.float32)

        cos_half_theta = np.sum(q0 * q1, axis=-1)

        neg_mask = cos_half_theta < 0
        q1 = q1.copy()
        q1[neg_mask] = -q1[neg_mask]

        cos_half_theta = np.abs(cos_half_theta)
        cos_half_theta = cos_half_theta[..., None]
        cos_half_theta = np.clip(cos_half_theta, -1.0, 1.0)

        half_theta = np.arccos(cos_half_theta)
        sin_half_theta = np.sqrt(np.maximum(1.0 - cos_half_theta * cos_half_theta, 0.0))

        sin_safe = np.maximum(sin_half_theta, eps)

        ratioA = np.sin((1.0 - weight) * half_theta) / sin_safe
        ratioB = np.sin(weight * half_theta) / sin_safe

        new_q = ratioA * q0 + ratioB * q1

        new_q = np.where(np.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = np.where(np.abs(cos_half_theta) >= 1.0, q0, new_q)

        return new_q
    
    def quat_normalize(self, quat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        quat = self.to_float32(quat)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        return quat / np.clip(norm, eps, None)
    
    def to_float32(self, array_like: Any) -> np.ndarray:
        arr = np.asarray(array_like)
        return arr.astype(np.float32, copy=False) if arr.dtype != np.float32 else arr
    

if __name__ == "__main__":
    motion_lib_cfg = MotionLibCfg()
    motion_fps_randomize = MotionLibOffline(motion_lib_cfg)
    motion_fps_randomize.randomization_motion()

