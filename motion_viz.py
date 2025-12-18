
import os
import time
import json
import joblib
import mujoco
import mujoco.viewer
import numpy as np
from typing import Dict, Any, List, Optional
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from motion.smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

HUMAN_BODY_LINKS = [
    'pelvis', # 0
                   
    'left_hip_link', 'left_knee_link', 'left_foot_link', 'left_toe_link', # 1, 2, 3, 4

    'right_hip_link', 'right_knee_link', 'right_foot_link', 'right_toe_link', # 5, 6, 7, 8

    'spine0_link', 'spine1_link', 'spine2_link', # 9, 10, 11

    'left_thorax_link', 'left_shoulder_link', 'left_elbow_link', 'left_wrist_link', # 12, 13, 14, 15

    'right_thorax_link', 'right_shoulder_link', 'right_elbow_link', 'right_wrist_link', # 16, 17, 18, 19

    'neck0_link', 'neck1_link', 'head_link' # 20, 21, 22
]

HUMAN_BODY_LINKS_PARENT_MAP = {
    "pelvis": "world",

    'left_hip_link': "pelvis",
    'left_knee_link': "left_hip_link",
    'left_foot_link': "left_knee_link",
    'left_toe_link': "left_foot_link",

    'right_hip_link': "pelvis",
    'right_knee_link': "right_hip_link",
    'right_foot_link': "right_knee_link",
    'right_toe_link': "right_foot_link",

    'spine0_link': "pelvis",
    'spine1_link': "spine0_link",
    'spine2_link': "spine1_link",

    'left_thorax_link': "spine2_link",
    'left_shoulder_link': "left_thorax_link",
    'left_elbow_link': "left_shoulder_link",
    'left_wrist_link': "left_elbow_link",

    'right_thorax_link': "spine2_link",
    'right_shoulder_link': "right_thorax_link",
    'right_elbow_link': "right_shoulder_link",
    'right_wrist_link': "right_elbow_link",

    'neck0_link': "spine2_link",
    'neck1_link': "neck0_link",
    'head_link': "neck1_link",
}

HUMAN_KEYPOINTS_LINKS = ['pelvis', 
                         
    'left_hip_link', 'left_knee_link', 'left_foot_link',

    'right_hip_link', 'right_knee_link', 'right_foot_link',

    'left_shoulder_link', 'left_elbow_link', 'left_wrist_link', 

    'right_shoulder_link', 'right_elbow_link', 'right_wrist_link', 
]

HUMAN_KEYPOINTS_PARENT_MAP = {
    "pelvis": "world",

    'left_hip_link': "pelvis",
    'left_knee_link': "left_hip_link",
    'left_foot_link': "left_knee_link",

    'right_hip_link': "pelvis",
    'right_knee_link': "right_hip_link",
    'right_foot_link': "right_knee_link",

    'left_shoulder_link': "pelvis",
    'left_elbow_link': "left_shoulder_link",
    'left_wrist_link': "left_elbow_link",

    'right_shoulder_link': "pelvis",
    'right_elbow_link': "right_shoulder_link",
    'right_wrist_link': "right_elbow_link",
}

class Data_Loader:

    @staticmethod
    def load_motion_data(motion_path, data_type="pkl"):
        
        data = {}

        if data_type == "pkl":

            if motion_path.endswith('.pkl'):
                key = motion_path[:-4]
                motion_data = joblib.load(motion_path)
                data[key] = motion_data
                return data
            
            files = [f for f in os.listdir(motion_path) if f.endswith('.pkl')]
            
            for fname in files:
                key = fname[:-4]
                try:
                    motion_data = joblib.load(os.path.join(motion_path, fname))
                    data[key] = motion_data
                except Exception as e:
                    print(f"Error loading {fname}: {e}")

        elif data_type == "json":

            if motion_path.endswith('.json'):
                key = motion_path[:-4]
                with open(motion_path, 'r') as f:
                    motion_data = json.load(f)
                data[key] = motion_data
                return data
            
            files = [f for f in os.listdir(motion_path) if f.endswith('.json')]
            
            for file in files:
                key = file[:-4]
                try:
                    motion_file = os.path.join(motion_path, file)
                    with open(motion_file, 'r') as f:
                        motion_data = json.load(f)
                    data[key] = motion_data
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        return data
    
class MotionVisualizer:

    def __init__(self, data: Dict[str, Dict[str, Any]], bad_dir: str, fps: float = 50.0):
        self.data = data
        self.bad_dir = bad_dir

        self.play_fps = float(fps)
        self.is_paused = False
        self.is_closed = False
        self._switch_request = 0

        self.show_rotation = True

        self.keys: List[str] = list(self.data.keys())
        self.data_len = len(self.keys)
        self.curr_data_id = 0

        self.curr_key: Optional[str] = None
        self.xyz: Optional[np.ndarray] = None   # (T,J,3)
        self.wxyz: Optional[np.ndarray] = None  # (T,J,4)
        self.num_frames = 0
        self.current_frame = 0

        self.keypoints_indices = [HUMAN_BODY_LINKS.index(l) for l in HUMAN_KEYPOINTS_LINKS]

        self.body_links_parent_indices, self.keypoints_parent_indices = self._compute_parent_indices()
        self.body_bone_pairs = [(i, p) for i, p in enumerate(self.body_links_parent_indices) if p != -1]
        self.kp_bone_pairs = [(i, p) for i, p in enumerate(self.keypoints_parent_indices) if p != -1]

        self.fig = plt.figure(figsize=(16, 8))
        self.ax_full = self.fig.add_subplot(121, projection="3d")
        self.ax_kp = self.fig.add_subplot(122, projection="3d")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._setup_axis(self.ax_full)
        self._setup_axis(self.ax_kp)

        self._init_plot()

        if self.data_len > 0:
            self._load_current_data()
            self._render(frame_idx=0, force_draw=True)

    def _on_close(self, event):
        self.is_closed = True

    def _on_key(self, event):
        k = (event.key or "").lower()

        if k == " ":
            self.is_paused = not self.is_paused

        elif k == "r":
            self.current_frame = 0

        elif k in ["n", "right"]:
            self._switch_request = +1

        elif k in ["p", "left"]:
            self._switch_request = -1

        elif k == "b":
            if self.curr_key is not None:
                os.makedirs(self.bad_dir, exist_ok=True)
                target = f"{self.bad_dir}/{self.curr_key}.pkl"
                joblib.dump(self.data[self.curr_key], target)

        elif k in ["+", "="]:
            self._set_play_fps(self.play_fps * 1.25)
        elif k in ["-", "_"]:
            self._set_play_fps(self.play_fps / 1.25)
        elif k == "0":
            self._set_play_fps(50.0)

        elif k in [".", ">"]:
            if self.num_frames > 0:
                self.is_paused = True
                self.current_frame = (self.current_frame + 1) % self.num_frames
                self._render(self.current_frame, force_draw=True)
        elif k in [",", "<"]:
            if self.num_frames > 0:
                self.is_paused = True
                self.current_frame = (self.current_frame - 1) % self.num_frames
                self._render(self.current_frame, force_draw=True)

        elif k == "t":
            self.show_rotation = not self.show_rotation

            if self.num_frames > 0:
                self._render(self.current_frame, force_draw=True)

        elif k in ["q", "escape"]:
            self.is_closed = True
            plt.close(self.fig)

    def _set_play_fps(self, new_fps: float):
        new_fps = float(new_fps)
        new_fps = max(1.0, min(new_fps, 120.0))
        self.play_fps = new_fps

    def _load_current_data(self):
        if self.data_len == 0:
            self.curr_key = None
            self.xyz = None
            self.wxyz = None
            self.num_frames = 0
            self.current_frame = 0
            return

        self.curr_data_id %= self.data_len
        self.curr_key = self.keys[self.curr_data_id]
        curr = self.data[self.curr_key]

        self.xyz = np.asarray(curr["body_pos"])   # (T,J,3)
        self.wxyz = np.asarray(curr["body_rot"])  # (T,J,4) wxyz

        self.num_frames = int(self.xyz.shape[0])
        self.current_frame = 0

    def _compute_parent_indices(self):
        body_parent = []
        for link in HUMAN_BODY_LINKS:
            parent = HUMAN_BODY_LINKS_PARENT_MAP.get(link, "world")
            while parent not in HUMAN_BODY_LINKS and parent != "world":
                parent = HUMAN_BODY_LINKS_PARENT_MAP.get(parent, "world")
            body_parent.append(-1 if parent == "world" else HUMAN_BODY_LINKS.index(parent))

        kp_parent = []
        for link in HUMAN_KEYPOINTS_LINKS:
            parent = HUMAN_KEYPOINTS_PARENT_MAP.get(link, "world")
            while parent not in HUMAN_KEYPOINTS_LINKS and parent != "world":
                parent = HUMAN_KEYPOINTS_PARENT_MAP.get(parent, "world")
            kp_parent.append(-1 if parent == "world" else HUMAN_KEYPOINTS_LINKS.index(parent))

        return body_parent, kp_parent

    def _setup_axis(self, ax):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(False)

    def _init_plot(self):
        # full body
        self.full_bones = Line3DCollection([], colors="black", linewidths=1.5, alpha=0.6)
        self.full_x = Line3DCollection([], colors="red", linewidths=1.2)
        self.full_y = Line3DCollection([], colors="green", linewidths=1.2)
        self.full_z = Line3DCollection([], colors="blue", linewidths=1.2)
        self.full_pts = self.ax_full.scatter([], [], [], c="red", s=15)

        self.ax_full.add_collection3d(self.full_bones)
        self.ax_full.add_collection3d(self.full_x)
        self.ax_full.add_collection3d(self.full_y)
        self.ax_full.add_collection3d(self.full_z)

        # keypoint body
        self.kp_bones = Line3DCollection([], colors="black", linewidths=1.5, alpha=0.6)
        self.kp_x = Line3DCollection([], colors="red", linewidths=1.2)
        self.kp_y = Line3DCollection([], colors="green", linewidths=1.2)
        self.kp_z = Line3DCollection([], colors="blue", linewidths=1.2)
        self.kp_pts = self.ax_kp.scatter([], [], [], c="orange", s=15)

        self.ax_kp.add_collection3d(self.kp_bones)
        self.ax_kp.add_collection3d(self.kp_x)
        self.ax_kp.add_collection3d(self.kp_y)
        self.ax_kp.add_collection3d(self.kp_z)

        self._set_rotation_visible(True)

    def _set_rotation_visible(self, visible: bool):
        for col in [self.full_x, self.full_y, self.full_z, self.kp_x, self.kp_y, self.kp_z]:
            col.set_visible(visible)

    def _set_limits_and_title(self, ax, title: str, center: np.ndarray):
        ax.set_title(title)
        ax.set_xlim([center[0] - 0.8, center[0] + 0.8])
        ax.set_ylim([center[1] - 0.8, center[1] + 0.8])
        ax.set_zlim([0.0, 1.8])

    def _update_bones(self, bones_collection: Line3DCollection, pose_xyz: np.ndarray, pairs: List[tuple]):
        if len(pairs) == 0:
            bones_collection.set_segments([])
            return
        i_idx = np.array([i for i, _ in pairs], dtype=np.int64)
        p_idx = np.array([p for _, p in pairs], dtype=np.int64)
        segs = np.stack([pose_xyz[i_idx], pose_xyz[p_idx]], axis=1)  # (nb,2,3)
        bones_collection.set_segments(segs)

    def _update_points(self, scatter, pose_xyz: np.ndarray):
        scatter._offsets3d = (pose_xyz[:, 0], pose_xyz[:, 1], pose_xyz[:, 2])

    def _update_rotations(self, xcol, ycol, zcol, pose_xyz: np.ndarray, quat_wxyz: np.ndarray, scale: float = 0.08):
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        rot_mats = R.from_quat(quat_xyzw).as_matrix()  # (N,3,3)

        x_ends = pose_xyz + scale * rot_mats[:, :, 0]
        y_ends = pose_xyz + scale * rot_mats[:, :, 1]
        z_ends = pose_xyz + scale * rot_mats[:, :, 2]

        xcol.set_segments(np.stack([pose_xyz, x_ends], axis=1))
        ycol.set_segments(np.stack([pose_xyz, y_ends], axis=1))
        zcol.set_segments(np.stack([pose_xyz, z_ends], axis=1))

    def _redraw(self, force_draw: bool):
        if force_draw:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()

        try:
            self.fig.canvas.flush_events()
        except Exception:
            plt.pause(0)

    def _render(self, frame_idx: int, force_draw: bool = False):
        if self.xyz is None or self.wxyz is None or self.curr_key is None or self.num_frames <= 0:
            self._redraw(force_draw=True)
            return

        frame_idx %= self.num_frames

        xyz = self.xyz[frame_idx]      # (J,3)
        wxyz = self.wxyz[frame_idx]    # (J,4)

        center = np.mean(xyz, axis=0)
        title = (
            f"[{self.curr_data_id+1}/{self.data_len}] {self.curr_key} | "
            f"Full | Frame {frame_idx}/{self.num_frames} | "
            f"fps={self.play_fps:.2f} | rotation={'on' if self.show_rotation else 'off'}"
        )
        self._set_limits_and_title(self.ax_full, title, center)
        self._update_bones(self.full_bones, xyz, self.body_bone_pairs)
        self._update_points(self.full_pts, xyz)

        kp_xyz = xyz[self.keypoints_indices]
        kp_wxyz = wxyz[self.keypoints_indices]
        kp_center = np.mean(kp_xyz, axis=0)
        kp_title = f"[{self.curr_data_id+1}/{self.data_len}] {self.curr_key} | Keypoints | Frame {frame_idx}/{self.num_frames}"
        self._set_limits_and_title(self.ax_kp, kp_title, kp_center)
        self._update_bones(self.kp_bones, kp_xyz, self.kp_bone_pairs)
        self._update_points(self.kp_pts, kp_xyz)

        self._set_rotation_visible(self.show_rotation)
        if self.show_rotation:
            self._update_rotations(self.full_x, self.full_y, self.full_z, xyz, wxyz)
            self._update_rotations(self.kp_x, self.kp_y, self.kp_z, kp_xyz, kp_wxyz)

        self._redraw(force_draw=force_draw)

    def run(self):
        if self.data_len == 0:
            raise ValueError("data is empty")

        plt.ion()
        try:
            plt.show(block=False)
        except Exception:
            pass

        next_tick = time.perf_counter()
        dropped_frames = 0

        while not self.is_closed:
            if self._switch_request != 0:
                self.curr_data_id = (self.curr_data_id + self._switch_request) % self.data_len
                self._switch_request = 0
                self.is_paused = False
                self._load_current_data()
                self.current_frame = 0
                self._render(0, force_draw=True)

                next_tick = time.perf_counter() + (1.0 / self.play_fps)
                continue

            if self.is_paused:
                self._redraw(force_draw=False)  #####保持绘制这一帧画面,不要更新帧,但是可以响应GUI窗口重置和切换事件,保持短暂休眠
                time.sleep(0.01)
                next_tick = time.perf_counter() + (1.0 / self.play_fps) #######next_tick重置为现在加上未来一帧时间
                continue

            if self.num_frames <= 0:
                self._redraw(force_draw=False)  #####没有数据,防止空转
                time.sleep(0.01)
                next_tick = time.perf_counter() + (1.0 / self.play_fps) #######next_tick重置为现在加上未来一帧时间
                continue

            period = 1.0 / max(self.play_fps, 1e-6)  #####每帧间隔
            now = time.perf_counter()

            if now < next_tick:
                time.sleep(next_tick - now)  ######避免渲染太快,绘制快于数据帧率,保持数据帧率
                now = time.perf_counter()

            if now - next_tick > period:
                behind = now - next_tick  ######检测now是否在next之后,检测渲染是否过慢,导致延后
                skip = int(behind // period) ######计算渲染过慢导致的数据延后了多少帧
                if skip > 0:
                    self.current_frame = (self.current_frame + skip) % self.num_frames   ######追赶延后的帧数,保持数据帧率

                    dropped_frames += skip
                    next_tick += skip * period ######将 next_tick向前推进 skip * period，对齐到“本该播放的那一帧”的时间
            self._render(self.current_frame, force_draw=False)
            self.current_frame = (self.current_frame + 1) % self.num_frames

            next_tick += period

        plt.ioff()

class RobotVisualizer:
    def __init__(self, data, humanoid_xml, dt=1/50):
        self.humanoid_xml = humanoid_xml
        self.dt = dt
        self.motion_data = data
        self.motion_keys = list(self.motion_data.keys())
        self.motion_id = 0
        self.time_step = 0
        self.paused = False
        self.sk_tree = SkeletonTree.from_mjcf(humanoid_xml)
        self.model = mujoco.MjModel.from_xml_path(humanoid_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt

    def add_visual_sphere(self, scene, position, radius, rgba):

        if scene.ngeom >= scene.maxgeom:
            return
        
        geom = scene.geoms[scene.ngeom]

        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0, 0]),
            pos=position.astype(np.float32),
            mat=np.eye(3).flatten(),
            rgba=rgba.astype(np.float32)
        )
        scene.ngeom += 1
        
    def add_visual_capsule(self, scene, point1, point2, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            rgba.astype(np.float32)
        )
        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            *point1, *point2
        )
        scene.ngeom += 1

    def key_callback(self, keycode):
        key = chr(keycode)
        if key == "R":
            print("Resetting time step.")
            self.time_step = 0
        elif key == " ":
            self.paused = not self.paused
            print("Paused" if self.paused else "Unpaused")
        elif key == "P":
            self.motion_id = (self.motion_id + 1) % len(self.motion_keys)
            print(f"Switched to: {self.motion_keys[self.motion_id]}")
        elif key == "A":
            curr_key = self.motion_keys[self.motion_id]
            print("Abandon:", curr_key)
            os.makedirs("tmp/relabel", exist_ok=True)
            with open("tmp/relabel/abandoned.txt", "a") as f:
                f.write(curr_key + "\n")
        else:
            print("Unmapped key:", key)

    def update_viewer(self, viewer):
        curr_key = self.motion_keys[self.motion_id]
        curr_motion = self.motion_data[curr_key]
        curr_time = int(self.time_step / self.dt) % curr_motion['joint_pos'].shape[0]

        body_pos = curr_motion["body_pos"]
        body_rot = curr_motion["body_rot"]
        joint_pos = curr_motion["joint_pos"]
        root_trans = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        self.data.qpos[:3] = root_trans[curr_time]
        self.data.qpos[3:7] = root_rot[curr_time]
        self.data.qpos[7:] = joint_pos[curr_time]

        mujoco.mj_forward(self.model, self.data)
        joints = curr_motion['body_pos'][curr_time]
        for i, pos in enumerate(joints):
            viewer.user_scn.geoms[i].pos = pos

    def run(self):
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        ) as viewer:
            for _ in range(50):
                self.add_visual_capsule(
                    viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1])
                )
            while viewer.is_running():
                step_start = time.time()
                self.update_viewer(viewer)
                if not self.paused:
                    self.time_step += self.dt
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)

def main():
    motion_file = "Datasets/target_data/g1_29dof/X/251217"
    humanoid_xml = 'assets/robots/g1/g1_29dof.xml'
    bad_dir = "Datasets/target_data/g1/bad/amass"

    data = Data_Loader.load_motion_data(motion_file)
    Motion_viz = MotionVisualizer(data, bad_dir)
    RobotViz = RobotVisualizer(data, humanoid_xml)

    Motion_viz.run()
    # RobotViz.run()

if __name__ == "__main__":
    main()