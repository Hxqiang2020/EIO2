import os
import time
import joblib
import mujoco
import mujoco.viewer
import numpy as np
from typing import List, Dict

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
    def load_motion_data(motion_path):
        
        data = {}

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

        return data

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


class MotionVisualizer:

    def __init__(self, data: Dict):

        self.data = data
        
        self.is_paused = False
        self.is_closed = False

        self.keys = list(self.data.keys())
        self.data_len = len(self.keys)
        self.curr_data_id = 0

        self.curr_key = None
        self.xyz = None
        self.wxyz = None
        self.num_frames = 0
        self.current_frame = 0
        self._switch_request = 0

        self.fig = plt.figure(figsize=(16, 8))
        self.ax_full = self.fig.add_subplot(121, projection="3d")
        self.ax_kp = self.fig.add_subplot(122, projection="3d")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.keypoints_indices = [HUMAN_BODY_LINKS.index(l) for l in HUMAN_KEYPOINTS_LINKS]

        self._compute_parent_indices()

        if self.data_len > 0:
            self._load_current_data()

    def _on_close(self, event):
        self.is_closed = True

    def _compute_parent_indices(self):
        self.body_links_parent_indices = []
        for link in HUMAN_BODY_LINKS:
            parent = HUMAN_BODY_LINKS_PARENT_MAP.get(link, "world")
            while parent not in HUMAN_BODY_LINKS and parent != "world":
                parent = HUMAN_BODY_LINKS_PARENT_MAP.get(parent, "world")
            if parent == "world":
                self.body_links_parent_indices.append(-1)
            else:
                self.body_links_parent_indices.append(HUMAN_BODY_LINKS.index(parent))

        self.keypoints_parent_indices = []
        for link in HUMAN_KEYPOINTS_LINKS:
            parent = HUMAN_KEYPOINTS_PARENT_MAP.get(link, "world")
            while parent not in HUMAN_KEYPOINTS_LINKS and parent != "world":
                parent = HUMAN_KEYPOINTS_PARENT_MAP.get(parent, "world")
            if parent == "world":
                self.keypoints_parent_indices.append(-1)
            else:
                self.keypoints_parent_indices.append(HUMAN_KEYPOINTS_LINKS.index(parent))

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
            if hasattr(self, "file_path") and hasattr(self, "bad_dir"):
                import shutil
                file = self.file_path
                bad_dir = self.bad_dir
                bad_dir.mkdir(parents=True, exist_ok=True)
                target_path = bad_dir / file.name
                shutil.copy(file, target_path)

        elif k in ["q", "escape"]:
            self.is_closed = True
            plt.close(self.fig)

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
        curr_data = self.data[self.curr_key]

        self.xyz = np.asarray(curr_data["body_pos"])   # (T, J, 3)
        self.wxyz = np.asarray(curr_data["body_rot"])  # (T, J, 4)

        self.num_frames = int(self.xyz.shape[0])
        self.current_frame = 0

        self.ax_full.clear()
        self.ax_kp.clear()
        self.fig.canvas.draw_idle()

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
        title = f"[{self.curr_data_id+1}/{self.data_len}] {self.curr_key} | Full | Frame {frame_idx}/{self.num_frames}"
        self._set_axes_common(ax, title, center)

        self._draw_pose_and_rot(
            ax=ax,
            pose_xyz=current_xyz,
            quat_xyzw=current_wxyz[:, [1, 2, 3, 0]],
            parent_indices=self.body_links_parent_indices,
            point_color="red",
        )

    def _draw_keypoint_frame(self, frame_idx: int):
        ax = self.ax_kp
        ax.clear()

        full_xyz = self.xyz[frame_idx]          # (J, 3)
        full_wxyz = self.wxyz[frame_idx]        # (J, 4)
        kp_xyz = full_xyz[self.keypoints_indices]        # (K, 3)
        kp_wxyz = full_wxyz[self.keypoints_indices]      # (K, 4)

        center = np.mean(kp_xyz, axis=0)
        title = f"[{self.curr_data_id+1}/{self.data_len}] {self.curr_key} | Keypoints | Frame {frame_idx}/{self.num_frames}"
        self._set_axes_common(ax, title, center)

        self._draw_pose_and_rot(
            ax=ax,
            pose_xyz=kp_xyz,
            quat_xyzw=kp_wxyz[:, [1, 2, 3, 0]],
            parent_indices=self.keypoints_parent_indices,
            point_color="orange",
        )
      
    def run(self):
        
        if self.data_len == 0:
            raise ValueError("data is empty")

        plt.ion()
        try:
            while not self.is_closed:
                if self._switch_request != 0:
                    self.curr_data_id = (self.curr_data_id + self._switch_request) % self.data_len
                    self._switch_request = 0
                    self._load_current_data()
                    continue

                if self.num_frames <= 0:
                    plt.pause(0.1)
                    continue

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
        finally:
            plt.ioff()

def main():
    motion_file = "Datasets/source_data//g1/amass"
    humanoid_xml = 'assets/robots/g1/g1_29dof.xml'

    data = Data_Loader.load_motion_data(motion_file)

    Motion_viz = MotionVisualizer(data)
    RobotViz = RobotVisualizer(data, humanoid_xml)

    Motion_viz.run()
    # RobotViz.run()

if __name__ == "__main__":
    main()

