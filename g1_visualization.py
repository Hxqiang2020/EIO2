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

    def _load_motion_data(self, motion_dir):

        files = [f for f in os.listdir(motion_dir) if f.endswith('.pkl')]
        data = {}
        for fname in files:
            key = fname[:-4]
            try:
                d = joblib.load(os.path.join(motion_dir, fname))
                data[key] = d
            except Exception as e:
                print(f"Error loading {fname}: {e}")
        keys = list(data.keys())
        return data, keys

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
        curr_time = int(self.time_step / self.dt) % curr_motion['reset_joint_pos'].shape[0]

        joint_pos = curr_motion["reset_joint_pos"]
        root_trans = curr_motion['reset_root_trans']
        root_rot = curr_motion['reset_root_rot']
        
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
    motion_file = "Datasets/target_data/randomization_fps/A"
    humanoid_xml = 'assets/robots/g1/g1_29dof.xml'

    # motion_file = "Datasets/target_data/o1/test1"
    # humanoid_xml = 'assets/robots/o1/o1.xml'

    visualizer = MotionVisualizer(motion_file, humanoid_xml)
    visualizer.run()

if __name__ == "__main__":
    main()