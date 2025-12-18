import os
import sys
import json
import torch
import joblib
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.ndimage as filters
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple, Optional, Any
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import motion.smpl_sim.poselib.core.rotation3d as pRot

import time

MODEL_CONVENTIONS = {
    "mujoco_quat_order": "wxyz",
    "scipy_quat_order": "xyzw",
    "root_pos_idx": slice(0, 3),
    "root_quat_idx": slice(3, 7),
    "joints_start_idx": 7
}

try:
    from general_motion_retargeting import EGMR_ROOT_DIR
    from cfgs.g1_gmr_cfg import UE_links, UE_link_parents, UE_keypoints_links, UE_keypoints_parents
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import project dependencies. {e}")
    sys.exit(1)

def setup_logging(log_dir: Path, log_name: str = "pipeline"):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_name}_{timestamp}.log"

    logger = logging.getLogger("G1Pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    return logger, log_file

logger = logging.getLogger("G1Pipeline")

@dataclass
class GlobalConfig:
    # root_dir: Path = Path(EGMR_ROOT_DIR)
    REMOTE_PATH = "/home/hx/code/EIO2"
    root_dir: Path = Path(REMOTE_PATH)

    retarget_json: Path = root_dir / "cfgs/ik_configs/ue_to_g1.json"
    robot_xml: Path = root_dir / "assets/robots/g1/g1_29dof.xml"
    target_dir: Path = root_dir / "Datasets/target_data/g1/amass1"
    bad_dir: Path = root_dir / "Datasets/bad_data/g1/amass"
    src_dir: Path = root_dir / "Datasets/source_data/g1/amass1"

    # retarget_json: Path = root_dir / "cfgs/ik_configs/ue_to_o1.json"
    # robot_xml: Path = root_dir / "assets/robots/o1/o1.xml"
    # src_dir: Path = root_dir / "Datasets/source_data//o1/test1"
    # target_dir: Path = root_dir / "Datasets/target_data/o1/test1"
    # bad_dir: Path = root_dir / "Datasets/bad_data/o1/test"
    
    log_dir: Path = root_dir / "logs"

    human_height: float = 1.60
    fps_default: float = 50.0
    smooth_sigma: float = 2.0
    
    def validate(self):

        missing = []
        if not self.retarget_json.exists(): missing.append(f"Config: {self.retarget_json}"); logger.warning(f"Config not found: {self.retarget_json}")
        if not self.robot_xml.exists(): missing.append(f"XML: {self.robot_xml}"); logger.warning(f"XML not found: {self.robot_xml}")
        
        if missing:
            error_msg = "Missing critical resources:\n" + "\n".join(missing)
            logger.critical(error_msg)
            raise FileNotFoundError(error_msg)

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

class SkeletonStructure:
    def __init__(self, links: List[str], parents_map: Dict[str, str]):
        self.links = links
        self.parents_map = parents_map
        self.parent_indices = self._compute_parent_indices()
        
    def _compute_parent_indices(self) -> List[int]:
        indices = []
        for l in self.links:
            p = self.parents_map.get(l, "world")
            while p not in self.links and p != "world":
                p = self.parents_map.get(p, "world")
            
            if p == "world":
                indices.append(-1)
            else:
                indices.append(self.links.index(p))
        return indices #[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21]


class MotionVisualizer:
    def __init__(self, xyz, wxyz, skeleton, keypoint_skeleton, selected_indices, file_path, bad_dir, title=""):
        self.xyz = xyz  # (T, J, 3)
        self.wxyz = wxyz  # (T, J, 4) [w, x, y, z]
        self.skeleton = skeleton
        self.keypoint_skeleton = keypoint_skeleton
        self.selected_indices = np.asarray(selected_indices, dtype=np.int64)
        self.num_frames = len(self.xyz)
        self.title = title
        self.file_path = file_path
        self.bad_dir = bad_dir

        self.current_frame = 0
        self.is_paused = False
        self.is_closed = False

        self.fig = plt.figure(figsize=(16, 8))
        self.ax_full = self.fig.add_subplot(121, projection="3d")
        self.ax_kp = self.fig.add_subplot(122, projection="3d")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        logger.info(f"Visualizer started for {title}. Controls: [Space] Pause, [R] Reset, [Q] Quit")

    def _on_key(self, event):
        if event.key == " ":
            self.is_paused = not self.is_paused
        elif event.key.lower() == "r":
            self.current_frame = 0
            logger.info("Reset to frame 0")
        elif event.key.lower() == "b":
            import shutil
            file = self.file_path
            bad_dir = self.bad_dir
            bad_dir.mkdir(parents=True, exist_ok=True)
            target_path = bad_dir / file.name
            shutil.copy(file, target_path)
            logger.info(f"Saved bad motion to: {target_path}")
        elif event.key.lower() in ["q", "escape"]:
            self.is_closed = True
            plt.close(self.fig)
            logger.info("Visualizer closed.")

    def _set_axes_common(self, ax, title, center):
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim([center[0] - 0.8, center[0] + 0.8])
        ax.set_ylim([center[1] - 0.8, center[1] + 0.8])
        ax.set_zlim([0.0, 1.8])

    def _draw_pose_and_rot(self, ax, pose_xyz, quat_xyzw, parent_indices, point_color="red"):

        bone_segments = []
        for i, p in enumerate(parent_indices):
            if p != -1:
                bone_segments.append([pose_xyz[i], pose_xyz[p]])
        ax.add_collection3d(Line3DCollection(bone_segments, colors="black", linewidths=1.5, alpha=0.6))

        rot_mats = R.from_quat(quat_xyzw).as_matrix()  # (N,3,3)
        scale = 0.08
        x_ends = pose_xyz + scale * rot_mats[:, :, 0]
        y_ends = pose_xyz + scale * rot_mats[:, :, 1]
        z_ends = pose_xyz + scale * rot_mats[:, :, 2]

        x_segs = [[pose_xyz[i], x_ends[i]] for i in range(len(pose_xyz))]
        y_segs = [[pose_xyz[i], y_ends[i]] for i in range(len(pose_xyz))]
        z_segs = [[pose_xyz[i], z_ends[i]] for i in range(len(pose_xyz))]

        ax.add_collection3d(Line3DCollection(x_segs, colors="red", linewidths=1.2))
        ax.add_collection3d(Line3DCollection(y_segs, colors="green", linewidths=1.2))
        ax.add_collection3d(Line3DCollection(z_segs, colors="blue", linewidths=1.2))

        ax.scatter(pose_xyz[:, 0], pose_xyz[:, 1], pose_xyz[:, 2], c=point_color, s=15)
        # for i, (x, y, z) in enumerate(pose_xyz):
        #     ax.text(x, y, z, str(i), fontsize=12, ha='center', va='bottom', color='black')

    def _draw_frame(self, frame_idx: int):
        ax = self.ax_full
        ax.clear()

        current_xyz = self.xyz[frame_idx]       # (J, 3)
        current_wxyz = self.wxyz[frame_idx]     # (J, 4)

        center = np.mean(current_xyz, axis=0)
        self._set_axes_common(ax, f"{self.title} | Full | Frame {frame_idx}/{self.num_frames} (Space:Pause)", center)

        self._draw_pose_and_rot(
            ax=ax,
            pose_xyz=current_xyz,
            quat_xyzw=current_wxyz[:, [1, 2, 3, 0]],
            parent_indices=self.skeleton.parent_indices,
            point_color="red",
        )

    def _draw_keypoint_frame(self, frame_idx: int):
        ax = self.ax_kp
        ax.clear()

        full_xyz = self.xyz[frame_idx]          # (J, 3)
        full_wxyz = self.wxyz[frame_idx]        # (J, 4)
        kp_xyz = full_xyz[self.selected_indices]        # (K, 3)
        kp_wxyz = full_wxyz[self.selected_indices]      # (K, 4)

        center = np.mean(kp_xyz, axis=0)
        self._set_axes_common(ax, f"{self.title} | Keypoints | Frame {frame_idx}/{self.num_frames}", center,)

        self._draw_pose_and_rot(
            ax=ax,
            pose_xyz=kp_xyz,
            quat_xyzw=kp_wxyz[:, [1, 2, 3, 0]],
            parent_indices=self.keypoint_skeleton.parent_indices,
            point_color="orange",
        )

    def run(self):
        plt.ion()
        while not self.is_closed:
            if not self.is_paused:
                if self.current_frame >= self.num_frames:
                    self.current_frame = 0

                self._draw_frame(self.current_frame)
                self._draw_keypoint_frame(self.current_frame)

                self.current_frame = (self.current_frame + 1) % self.num_frames
                plt.draw()
                plt.pause(0.01)
            else:
                plt.pause(0.1)

        plt.ioff()
        plt.show()

class PipelineController:
    def __init__(self, config: GlobalConfig):
        self.cfg = config
        self.cfg.validate()
        self.retargeter = None
        self.selected_indices = []

        _, self.log_path = setup_logging(self.cfg.log_dir)
        logger.info(f"Pipeline initialized. Logs writing to: {self.log_path}")
        try:
            self.cfg.validate()
        except Exception:
            sys.exit(1)

    def _init_retargeting(self):
        if self.retargeter is None:
            logger.info("Lazy loading Retargeter engine (this takes a few seconds)...")
            self.retargeter = GMR(
                ik_config_path=str(self.cfg.retarget_json),
                tgt_robot_xml_path=str(self.cfg.robot_xml),
                actual_human_height=self.cfg.human_height
            )
            with open(self.cfg.retarget_json) as f: data = json.load(f)
            # selected_links = list(data["human_scale_table"].keys())
            selected_links = UE_keypoints_links
            self.selected_indices = [UE_links.index(l) for l in selected_links]

    def process_dataset(self):
        self._init_retargeting()
        self.cfg.target_dir.mkdir(parents=True, exist_ok=True)
        
        all_src_files = list(self.cfg.src_dir.rglob("*.json"))
        processed_files = list(self.cfg.target_dir.rglob("*.pkl"))
        processed_keys = {p.stem for p in processed_files}
        
        to_process = [p for p in all_src_files if p.stem not in processed_keys]
        logger.info(f"Dataset Scan Summary: Total {len(all_src_files)}, Remaining {len(to_process)}")

        if not to_process:
            logger.info("All files up to date. Exiting.")
            return
        
        stats = {"success": 0, "failed": 0, "errors": []}
        pbar = tqdm(to_process, desc="Retargeting", unit="file")

        for file_path in pbar:
            pbar.set_postfix(current=file_path.stem)
            try:
                self._process_single_file(file_path)
                stats["success"] += 1
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append((file_path.name, str(e)))
                logger.error(f"FAILED: {file_path.name} | Error: {e}", exc_info=True)

        logger.info("="*40)
        logger.info(f"PROCESSING COMPLETE. Success: {stats['success']}, Failed: {stats['failed']}")
        if stats["failed"] > 0:
            logger.warning(f"Check log file for details: {self.log_path}")
        
    def process_dataset_distribution(self):
        self._init_retargeting()
        self.cfg.target_dir.mkdir(parents=True, exist_ok=True)
        
        all_src_files = list(self.cfg.src_dir.rglob("*.json"))
        len_files = len(all_src_files)

        for i in range(len_files):
            print(f"{i}/{len_files}")
            try:
                time.sleep(0.1)
                self._process_single_file(all_src_files[i], del_current_file=True)
            except:
                continue

    def _process_single_file(self, file_path: Path, del_current_file=False):
        with open(file_path, 'r') as f:
            motion_data = json.load(f)
        if del_current_file:
            os.remove(file_path)
            
        fps = motion_data.get("mocap framrate", self.cfg.fps_default)
        dt = 1.0 / fps
        T = len(motion_data["poses"])

        human_motion = np.array(motion_data["poses"]).reshape(T, -1, 7)
        keypoints_subset = human_motion[:, self.selected_indices, :]

        body_pos = human_motion[:, :, :3].copy()
        body_pos[:, :, :2] = body_pos[:, :, :2] - body_pos[:1, :1, :2]
        body_pos[:, :, 2] = body_pos[:, :, 2] - np.min(body_pos[:, :, 2]) + 0.05

        body_rot = human_motion[:, :, 3:].copy()
        body_rot = MathEngine.quat_normalize(body_rot)

        g1_qpos_list = []
        for i in range(T):
            _, qpos = self.retargeter.retarget(keypoints_subset[i])
            g1_qpos_list.append(qpos)
            
        g1_qpos = np.stack(g1_qpos_list)
        start_idx = MODEL_CONVENTIONS["joints_start_idx"]
        joint_pos = g1_qpos[:, start_idx:]
        joint_vel = MathEngine._compute_dof_vel(joint_pos, dt)
        
        body_lin_vel = MathEngine._compute_velocity(body_pos, dt)
        body_ang_vel = MathEngine._compute_angular_velocity(body_rot, dt)

        relabel_root_rot = g1_qpos[:, 3:7]
        relabel_root_trans = g1_qpos[:, :3]
        relabel_root_trans[:, :2] -= relabel_root_trans[0, :2]
        
        target_data = {
            "body_pos": body_pos.astype(np.float32),
            "body_rot": body_rot.astype(np.float32),
            "body_lin_vel": body_lin_vel.astype(np.float32),
            "body_ang_vel": body_ang_vel.astype(np.float32),
            "reset_joint_pos": joint_pos.astype(np.float32),
            "reset_joint_vel": joint_vel.astype(np.float32),
            "reset_root_rot": relabel_root_rot.astype(np.float32),
            "reset_root_trans": relabel_root_trans.astype(np.float32),
            "robot": "unitree.g1_29dof",
        }
        
        save_path = self.cfg.target_dir / f"{file_path.stem}.pkl"
        joblib.dump(target_data, save_path)

    def visualize_file(self, file_path_str: str):
        self._init_retargeting()
        bad_dir = self.cfg.bad_dir
        file_path = Path(file_path_str)
        if not file_path.exists():
            file_path = self.cfg.src_dir / file_path_str
            if not file_path.exists():
                logger.error(f"File not found: {file_path_str}")
                return

        logger.info(f"Loading visualization for: {file_path}")

        with open(file_path, 'r') as f: motion_data = json.load(f)
        poses = np.array(motion_data["poses"]).reshape(len(motion_data["poses"]), -1, 7)
        xyz = poses[:, :, :3]
        wxyz = poses[:, :, 3:]
        wxyz = MathEngine.quat_normalize(wxyz)
        
        skeleton = SkeletonStructure(UE_links, UE_link_parents)
        keypoint_skeleton = SkeletonStructure(UE_keypoints_links, UE_keypoints_parents)
        viz = MotionVisualizer(xyz, wxyz, skeleton, keypoint_skeleton, self.selected_indices, file_path, bad_dir, title=file_path.stem)
        viz.run()

def main():
    parser = argparse.ArgumentParser(description="G1 Motion Retargeting Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands: process | viz")
    subparsers.add_parser("process", help="Batch process dataset")
    subparsers.add_parser("distribution", help="distribution process dataset")
    parser_viz = subparsers.add_parser("viz", help="Visualize motion")
    parser_viz.add_argument("file", type=str, help="Filename")
    
    if len(sys.argv) == 1:

        # DEBUG_MODE = "viz" 
        DEBUG_MODE = "distribution"
        # DEBUG_MODE = "process"
        DEBUG_FILE = ""

        logger.warning(f"No CLI args found. Switching to VS Code Debug Mode: [{DEBUG_MODE}]")
        
        if DEBUG_MODE == "viz": sys.argv.extend(["viz", DEBUG_FILE])
        elif DEBUG_MODE == "process": sys.argv.extend(["process"])
        elif DEBUG_MODE == "distribution": sys.argv.extend(["distribution"])

    args = parser.parse_args()
    
    config = GlobalConfig()
    controller = PipelineController(config)
    
    if args.command == "process": controller.process_dataset()
    elif args.command == "distribution": controller.process_dataset_distribution()
    elif args.command == "viz":
        for i in range(50):
            all_files =os.listdir(config.src_dir)
            file_num = len(os.listdir(config.src_dir))
            if file_num == 0:
                print("No files to visualize.")
                return
            elif file_num < 50:
                random_idx = i % file_num
            else:
                random_idx = np.random.randint(0, file_num)
            file = all_files[random_idx]
            file_name = os.path.basename(file)
            controller.visualize_file(file_name)

    else: parser.print_help()

if __name__ == "__main__":
    main()