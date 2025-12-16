import os
import os.path as osp
import time
import pickle
import joblib
from joblib.numpy_pickle import _unpickle

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from motion.smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
body_names = ['pelvis',
            'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
            'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 
            'waist_yaw_link', 'waist_roll_link', 'waist_pitch_link', 
            'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
            'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']

tracked_bodies = ['left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link',
                  
                'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_roll_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_roll_link']

class MotionVisualizer:
    def __init__(self, motion_path, humanoid_xml, dt=1/30):
        self.device = torch.device("cpu")
        self.humanoid_xml = humanoid_xml
        self.dt = dt
        self.motion_data, self.motion_keys = self._load_motion_data(motion_path)
        self.motion_id = 0
        self.time_step = 0
        self.paused = False
        self.sk_tree = SkeletonTree.from_mjcf(humanoid_xml)
        self.model = mujoco.MjModel.from_xml_path(humanoid_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt

        self.tracked_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in tracked_bodies]

    def _is_pkl_file_corrupted(self, file_path):
        return False
        # try:
        #     with open(file_path, 'rb') as f:
        #         _unpickle(f, filename=file_path, mmap_mode=None)
        #     return False
        # except Exception:
        #     return True

    def _clean_motion_files(self, motion_dir):
        files = [f for f in os.listdir(motion_dir) if f.endswith('.pkl')]
        valid = []
        for f in files:
            path = osp.join(motion_dir, f)
            if self._is_pkl_file_corrupted(path):
                print(f"Corrupted: {f}, deleting...")
                os.remove(path)
            else:
                valid.append(f)
        return valid

    def _load_motion_data(self, motion_path):
        if motion_path.endswith('.pkl'):
            data = joblib.load(motion_path)
            keys = list(data.keys())
            return data, keys
        print("Validating and loading motion files...")
        files = self._clean_motion_files(motion_path)
        data = {}
        for fname in files:
            key = fname[:-4]
            try:
                d = joblib.load(osp.join(motion_path, fname))
                data[key] = d
            except Exception as e:
                print(f"Error loading {fname}: {e}")
        keys = list(data.keys())
        return data, keys

    def add_visual_sphere(self, scene, position, radius, rgba):
        """添加球体几何图形到场景"""
        if scene.ngeom >= scene.maxgeom:
            return
        
        geom = scene.geoms[scene.ngeom]
        # 初始化几何体为球体
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0, 0]),  # 球体半径
            pos=position.astype(np.float32),  # 位置
            mat=np.eye(3).flatten(),  # 单位矩阵
            rgba=rgba.astype(np.float32)  # 颜色
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

        # self.data.qpos[:3] = curr_motion['relabel_root_trans'][curr_time]
        # self.data.qpos[3:7] = curr_motion['relabel_root_rot'][curr_time]

        # body_pos = curr_motion["body_pos"]
        # body_rot = curr_motion["body_rot"]
        # joint_pos = curr_motion["joint_pos"]
        # root_trans = body_pos[:, 0, :]
        # root_trans[:, :2] -= root_trans[0, :2]
        # root_rot = body_rot[:, 0, :]

        # self.data.qpos[:3] = root_trans[curr_time]
        # self.data.qpos[3:7] = root_rot[curr_time][[3, 0, 1, 2]]
        # self.data.qpos[7:] = joint_pos[curr_time]

        self.data.qpos[:3] = curr_motion['root_trans'][curr_time]
        self.data.qpos[3:7] = curr_motion['root_rot'][curr_time]
        self.data.qpos[7:] = curr_motion['joint_pos'][curr_time]

        mujoco.mj_forward(self.model, self.data)
        joints = curr_motion['body_pos'][curr_time]
        for i, pos in enumerate(joints):
            viewer.user_scn.geoms[i].pos = pos

        # robot_body_pos = self.data.xpos
        
        # # 清空之前的几何体
        # viewer.user_scn.ngeom = 0
        
        # for body_name in tracked_bodies:
        #     body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        #     if body_id != -1 and body_id < len(robot_body_pos):
        #         # 添加红色球体表示机器人本体关节
        #         if body_id < 18:
        #             color = np.array([0.0, 0.0, 1.0, 1.0])  # 蓝色，完全不透明
        #         else:
        #             color = np.array([0.0, 1.0, 0.0, 1.0])  # 绿色，完全不透明
                    
        #         self.add_visual_sphere(
        #             viewer.user_scn,
        #             robot_body_pos[body_id],
        #             0.06,  # 红点半径
        #             color  # 红色，完全不透明
        #         )
        
        # # 绘制机器人骨架
        # if hasattr(self, 'show_skeleton') and self.show_skeleton:
        #     self.draw_robot_skeleton(viewer)

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
    # motion_file = "Datasets/target_data/G1/test3"
    # motion_file = "Datasets/target_data/G1/G1/Amass"
    motion_file = "/home/hx/code/EMAN/sample_data/amass"
    # motion_file = "data/figure_data/t02ac"
    # motion_file = "data/T2M/With_Hand/0-motion_after_recover1.pkl"
    humanoid_xml = 'assets/robots/g1/g1_skeleton.xml'
    visualizer = MotionVisualizer(motion_file, humanoid_xml)
    visualizer.run()

if __name__ == "__main__":
    main()