
import numpy as np
import os
import yaml
from tqdm import tqdm
import time
import os.path as osp

from motion.utils import torch_utils
import joblib
import torch
import torch.multiprocessing as mp
# import multiprocessing as mp
import copy
import gc

from scipy.spatial.transform import Rotation as sRot
import random
from motion.utils.flags import flags
from motion.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode
from motion.utils.torch_humanoid_batch import Humanoid_Batch
from easydict import EasyDict
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)



USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

from torch.utils.data import DataLoader, Dataset

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


class MotionLibReal(MotionLibBase):

    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg = motion_lib_cfg)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg.robot)
        self.humanoid_type = motion_lib_cfg.humanoid_type
        return
    
    @staticmethod
    def fix_trans_height(pose_aa, trans, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        
        with torch.no_grad():
            
            mesh_obj = mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff
            
            return trans, height_diff
        
    def _load_motions(self, max_cached_motions, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1, target_heading=None):
        # load motion load the same number of motions as there are skeletons (humanoids)
        # if "gts" in self.__dict__:
        #     del self.gts, self.grs, self.lrs, self.grvs, self.gravs, self.gavs, self.gvs, self.dvs
        #     del self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa
        #     if "gts_t" in self.__dict__:
        #         self.gts_t, self.grs_t, self.gvs_t
        #     if flags.real_traj:
        #         del self.q_gts, self.q_grs, self.q_gavs, self.q_gvs

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        gc.collect()
        torch.cuda.empty_cache()

        total_len = 0.0
        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=min(num_motion_to_load, max_cached_motions), replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions ).to(self._device)
        
        LOAD_ONCE = True if (len(sample_idxes) >= self._num_unique_motions) else False
        if LOAD_ONCE: sample_idxes = torch.arange(self._num_unique_motions).to(self._device)

        # import ipdb; ipdb.set_trace()
        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(self._curr_motion_ids, num_classes = self._num_unique_motions).to(self._device)  # Testing for obs_v5
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        # print("\n****************************** Current motion keys ******************************")
        # print(f"LOAD_ONCE: {LOAD_ONCE}. Sampling: {len(sample_idxes)}/{len(self.curr_motion_keys)}.")
        # print("Sampling motion:", sample_idxes[:3], ".....")
        # if len(sample_idxes) <= 3:
        #     print(self.curr_motion_keys)
        # else:
        #     print(self.curr_motion_keys[:3], ".....")
        # print("*********************************************************************************\n")

        class MotionDataset(Dataset):
            def __init__(self, jobs, fun):
                self.jobs = jobs
                self.fun = fun

            def __len__(self):
                return len(self.jobs)

            def __getitem__(self, idx):
                args = self.jobs[idx]
                curr_id, curr_file, curr_motion = self.fun(*args, None, idx)

                c = 0
                while curr_motion is None:
                    l = self.__len__()
                    args = self.jobs[(idx + np.random.randint(l)) % l]
                    _, curr_file, curr_motion = self.fun(*args, None, idx)
                    c += 1
                    if c > 10: 
                        assert False
                return (curr_id, curr_file, curr_motion)
        
        def collate_fn(batch):
            collated_res = {}
            for sample in batch:
                curr_id, curr_file, curr_motion = sample
                collated_res[curr_id] = (curr_file, curr_motion)
            return collated_res

        torch.multiprocessing.set_sharing_strategy('file_system')

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]

        # torch.set_num_threads(1)
        # manager = mp.Manager()
        # queue = manager.Queue()
        # num_jobs = min(mp.cpu_count(), 64)

        # if num_jobs <= 16 or not self.multi_thread:
        #     num_jobs = 1
        # if flags.debug:
        #     num_jobs = 1
        
        # num_gpus = torch.cuda.device_count()
        # num_jobs = min(mp.cpu_count(), 8//num_gpus)
        # num_jobs = max(num_jobs, 2)
        num_jobs = mp.cpu_count() // 2
        
        res_acc = {}  # using dictionary ensures order of the results.
        jobs = motion_data_list
        chunk = 1 # np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        start_time = time.time()
        jobs = [(ids[i:i + chunk], jobs[i:i + chunk], skeleton_trees[i:i + chunk], gender_betas[i:i + chunk], self.fix_height, self.mesh_parsers, target_heading, max_len, self.m_cfg) for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]

        motion_dataset = MotionDataset(job_args, self.load_motion_with_skeleton)
        motion_dataloader = DataLoader(motion_dataset, batch_size=num_jobs, num_workers=num_jobs, shuffle=False, collate_fn=collate_fn)
        for batch in motion_dataloader: # tqdm
            res_acc.update(batch)
        # print("-->", len(res_acc))
        # print(f"Total execution time: {time.time() - start_time} seconds")

        if LOAD_ONCE:
            sample_idxes = sample_idxes.repeat(int(num_motion_to_load/len(sample_idxes)+1))[:num_motion_to_load]
            sample_idxes = sample_idxes[torch.randperm(len(sample_idxes))].clone()

            self._curr_motion_ids = None # sample_idxes
            self.one_hot_motions = None # torch.nn.functional.one_hot(self._curr_motion_ids, num_classes = self._num_unique_motions).to(self._device)
            self.curr_motion_keys = None # self._motion_data_keys[sample_idxes]
            self._sampling_batch_prob = None # self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # size = len(res_acc)
            # for i in range(size, num_motion_to_load):
            #     res_acc[i] = copy.deepcopy(res_acc[i % size])
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        # self._motion_file_data = []
        for f in range(len(res_acc)): # tqdm
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            # if "beta" in motion_file_data:
            #     _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
            #     _motion_bodies.append(curr_motion.gender_beta)
            # else:
            #     _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
            #     _motion_bodies.append(torch.zeros(17))

            # self._motion_file_data.append(motion_file_data)
            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion
            
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        #### self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        #### self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        #### self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        #### self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        #### self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        #### self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        #### self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
        #### if "global_translation_extend" in motions[0].__dict__:
        ####     self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
        ####     self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
        ####     self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
        ####     self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
        
        #### if flags.real_traj:
        ####     self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
        ####     self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
        ####     self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
        ####     self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = self.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        # print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        # return motions
    
    def load_motions(self, loading_ranks, **kwargs):
        import torch.distributed as dist
        if not dist.is_initialized():
            torch.manual_seed(time.time() % 10000 * 10000); self._load_motions(**kwargs)
            return
        
        rank = dist.get_rank()
        local_rank = dist.get_node_local_rank()
        local_world_size = torch.cuda.device_count()
        for current_local_rank in range(local_world_size):
            if (local_rank == current_local_rank) and (local_rank in loading_ranks):
                print(f"GPU {local_rank}|{rank}: Loading motions...")
                start_time = time.time()
                torch.manual_seed(time.time() % 10000 * 10000 + rank * 10000); self._load_motions(**kwargs)
                end_time = time.time()
                # print(f"GPU {local_rank}|{rank}: Finished loading in {end_time - start_time:.2f}s")
            dist.barrier()
        return
    
    def get_motion_length(self, motion_ids=None):
        if motion_ids is None: assert False
        return super().get_motion_length(motion_ids % self._num_motions)
    
    def sample_time(self, motion_ids, truncate_time=None):
        return super().sample_time(motion_ids % self._num_motions, truncate_time)

    def _get_motion_state(self, motion_ids, motion_times, offset=None):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        motion_ids = motion_ids % self._num_motions

        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        

        if "dof_pos" in self.__dict__: # H1 joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        motion_state_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = torch_utils.slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = torch_utils.slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel
            
        
        motion_state_dict.update({
            # "root_pos": rg_pos[..., 0, :].clone(),
            # "root_rot": rb_rot[..., 0, :].clone(),
            # "root_vel": body_vel[..., 0, :].clone(),
            # "root_ang_vel": body_ang_vel[..., 0, :].clone(),

            "joint_pos": dof_pos.clone(), # "dof_pos": dof_pos.clone(),
            "joint_vel": dof_vel.view(dof_vel.shape[0], -1), # "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            #### "motion_aa": self._motion_aa[f0l],
            #### "motion_bodies": self._motion_bodies[motion_ids],
            #### "motion_limb_weights": self._motion_limb_weights[motion_ids],
            
            "body_pos": rg_pos, #"rg_pos": rg_pos,
            "body_rot": rb_rot, # "rb_rot": rb_rot,
            "body_lin_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            
            #### "rg_pos_t": rg_pos_t,
            #### "rg_rot_t": rg_rot_t,
            #### "body_vel_t": body_vel_t,
            #### "body_ang_vel_t": body_ang_vel_t,
        })
        return motion_state_dict
    
    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_state_dict = self._get_motion_state(motion_ids, motion_times, offset)

        # if self.humanoid_type in ["g1_12dof", "g1_29dof"]:
        #     assert motion_dict["dof_pos"].shape[-1] == 29
        # elif self.humanoid_type == "hu_12dof":
        #     assert motion_dict["dof_pos"].shape[-1] == 32
        # else: raise ValueError(f"Unknown humanoid type: {self.humanoid_type}")
        
        # # dof_pos = motion_dict["dof_pos"]
        # # dof_pos = dof_pos[..., self.dof_axes]
        # # motion_dict["dof_pos"] = dof_pos

        # # dof_vel = motion_dict["dof_vel"]
        # # dof_vel = dof_vel[..., self.dof_axes]
        # # motion_dict["dof_vel"] = dof_vel
        
        # if self.humanoid_type == "g1_12dof" or self.humanoid_type == "hu_12dof":
        #     motion_dict["dof_pos"] = motion_dict["dof_pos"][:, :12]
        #     motion_dict["dof_vel"] = motion_dict["dof_vel"][:, :12]
        #     motion_dict["rg_pos"] = motion_dict["rg_pos"][:, :13]
        #     motion_dict["rb_rot"] = motion_dict["rb_rot"][:, :13]
        #     motion_dict["body_vel"] = motion_dict["body_vel"][:, :13]
        #     motion_dict["body_ang_vel"] = motion_dict["body_ang_vel"][:, :13]
        # elif self.humanoid_type == "g1_29dof":
        #     pass

        assert self.humanoid_type == "g1_29dof"
        
        return motion_state_dict

    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, mesh_parsers, target_heading, max_len, m_cfg, queue, pid):
        assert len(ids) == 1
        assert len(ids) == len(motion_data_list)
        
        folder = motion_data_list[0].split("/")
        assert ("sample_data" in folder) and (folder[-1][-4:] == ".pkl")
        folder = "/".join(folder[folder.index("sample_data") + 1: -1])

        curr_id = ids[0]
        curr_file = motion_data_list[0]

        if not isinstance(curr_file, dict) and osp.isfile(curr_file):
            key = curr_file.split("/")[-1].rsplit(".pkl", 1)[0]

            curr_file_fps_offset = 0 * (np.round(np.random.rand() * 4) - 2) # -10 ~ 10
            folder = folder + f"__FPS{int(30 + curr_file_fps_offset)}__"

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            if (max_len == -1) and (target_heading is None):
                try:
                    curr_motion = joblib.load(f"tmp/sample_data_cache/{folder}/{key}.pkl") #[key]
                    curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
                    return (curr_id, curr_file, curr_motion)
                except: pass
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            curr_file_motion = joblib.load(curr_file) #[key]
            assert curr_file_motion["fps"] == 30

        seq_len = curr_file_motion['root_trans_offset'].shape[0]

        if seq_len < 15: return (curr_id, curr_file, None)

        if max_len == -1 or seq_len < max_len:
            start, end = 0, seq_len
        else:
            start = random.randint(0, seq_len - max_len)
            end = start + max_len

        trans = to_torch(curr_file_motion['root_trans_offset'].astype(np.float32)[:]).clone()[start:end]
        pose_aa = to_torch(curr_file_motion['pose_aa'].astype(np.float32)[start:end]).clone()
        dt = 1. / (curr_file_motion['fps'] + curr_file_fps_offset)

        B, J, N = pose_aa.shape

        ##### ZL: randomize the heading ######
        # if (not flags.im_eval) and (not flags.test):
        #     # if True:
        #     random_rot = np.zeros(3)
        #     random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
        #     random_heading_rot = sRot.from_euler("xyz", random_rot)
        #     pose_aa = pose_aa.reshape(B, -1)
        #     pose_aa[:, :3] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec())
        #     trans = torch.matmul(trans, torch.from_numpy(random_heading_rot.as_matrix().T))
        ##### ZL: randomize the heading ######
        
        if not target_heading is None:
            start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
            heading_inv_rot = sRot.from_quat(torch_utils.calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ])))
            heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
            pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

            trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T.astype(np.float32)))
            trans[:, :2] = trans[:, :2] - trans[:1, :2] # zero the start x-y position
        
        trans, trans_fix = MotionLibReal.fix_trans_height(pose_aa, trans, mesh_parsers, fix_height_mode = fix_height)
        curr_motion = mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if (max_len == -1) and (target_heading is None):
            os.makedirs(f"tmp/sample_data_cache/{folder}/", exist_ok=True)
            joblib.dump(curr_motion, f"tmp/sample_data_cache/{folder}/{key}.pkl")
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
        return (curr_id, curr_file, curr_motion)

