import os
import sys
import time
import argparse
import pdb
import os.path as osp

import glob

sys.path.append(os.getcwd())

# from motion.smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
# from scipy.spatial.transform import Rotation as sRot
import joblib


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom - 1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             point1[0], point1[1], point1[2],
                             point2[0], point2[1], point2[2])


def key_call_back(keycode):
    pass


def check_self_collision(model, data):
    for i in range(data.ncon):
        contact = data.contact[i]
        # geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        # geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1])
        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2])

        penetration_depth = -contact.dist
        forces = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, forces)

        if body1_name == "world" or body2_name == "world":
            continue

        # Compute the magnitude of the normal force
        normal_force_magnitude = np.linalg.norm(forces[:3])
        if penetration_depth > 1e-6 and normal_force_magnitude > 0.1:
            # print(f"Collision detected between {body1_name} and {body2_name}, penetration depth:{penetration_depth}, force:{normal_force_magnitude:.2f}")

            return True
    return False

def check_pos_distance(pos1, pos2):
    difference_pos = pos1 - pos2 - pos1[0] + pos2[0] # [23, 3]
    difference_pos = np.linalg.norm(difference_pos, axis=-1)
    if difference_pos.max() > 0.5:
        return True
    return False

def main():

    # folder --> folder_clean
    folder = f"sample_data/motions2/unitree/g1_29dof/A"
    folder_clean = f"sample_data/motions2/unitree/g1_29dof/A_clean"
    os.makedirs(folder_clean, exist_ok=True)

    motions = glob.glob(os.path.join(folder, "*.pkl"))
    
    current_id, max_id = 0, len(motions)
    cleaned_frame, bad_frame = 0, 0
    bad_data = []

    humanoid_xml = '/home/hx/code/EIO2/assets/robots/g1/g1_29dof.xml'
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_model.geom_margin = 0.00001
    mj_model.opt.iterations = 100
    # mj_model.opt.timestep = 0.001

    mj_data = mujoco.MjData(mj_model)
    
    current_id = -1
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        while viewer.is_running() and current_id < max_id:
            current_id += 1

            if current_id >= max_id: 
                break

            curr_motion_key = os.path.basename(motions[current_id]).replace(".pkl", "")
            curr_motion = joblib.load(motions[current_id]) #[curr_motion_key]

            _len = curr_motion['body_pos'].shape[0]

            i = 0
            clean_index = -1
            clean_joint_pos = curr_motion['reset_joint_pos'].copy()
            clean_joint_vel = curr_motion['reset_joint_vel'].copy()
            uncleaned_list = []

            mujoco_body_lowest = []
            
            for i in range(_len):
                mj_data.qpos[:3] = curr_motion['reset_root_trans'][i]
                mj_data.qpos[3:7] = curr_motion['reset_root_rot'][i]
                mj_data.qpos[7:] = curr_motion['reset_joint_pos'][i]

                mujoco.mj_forward(mj_model, mj_data)
                # viewer.sync()

                mujoco_body_lowest.append(np.min(mj_data.xpos[1:, 2]))

                self_collision = check_self_collision(mj_model, mj_data)
                if self_collision:  # 有碰撞
                    if clean_index > -1:
                        # too_far = check_pos_distance(curr_motion['body_pos'][i], curr_motion['body_pos'][clean_index])
                        clean_joint_pos[i] = curr_motion['reset_joint_pos'][clean_index].copy()
                        clean_joint_vel[i] = curr_motion['reset_joint_vel'][clean_index].copy()
                    else:
                        uncleaned_list.append(i)
                    bad_frame += 1
                else:   # 无碰撞
                    if clean_index == -1:
                        for j in uncleaned_list:
                            # too_far = check_pos_distance(curr_motion['body_pos'][i], curr_motion['body_pos'][j])
                            clean_joint_pos[j] = curr_motion['reset_joint_pos'][i].copy()
                            clean_joint_vel[j] = curr_motion['reset_joint_vel'][i].copy()
                        uncleaned_list = []
                    clean_index = i
                    cleaned_frame += 1

                if i == _len - 1:
                    if clean_index == -1:   # all frame has collision
                        print(f"所有帧有碰撞: {curr_motion_key}")
                        bad_data.append(curr_motion_key)
                        break
                    
                    print(f"保存干净动作: {curr_motion_key}")
                    print(f"已处理: {current_id}/{max_id}")
                    
                    curr_motion['raw_joint_pos'] = curr_motion.pop('reset_joint_pos')
                    curr_motion['raw_joint_vel'] = curr_motion.pop('reset_joint_vel')

                    curr_motion['reset_joint_pos'] = clean_joint_pos.copy()
                    curr_motion['reset_joint_vel'] = clean_joint_vel.copy()

                    curr_motion['reset_root_rot'] = curr_motion.pop('reset_root_rot')
                    curr_motion['reset_root_trans'] = curr_motion.pop('reset_root_trans')
                    curr_motion['reset_root_trans'][:, 2] -= (np.array(mujoco_body_lowest).min() - 0.05)

                    _unique = "" # str(int(time.time()))
                    with open(os.path.join(folder_clean, curr_motion_key + _unique + ".pkl"), 'wb') as f:
                        joblib.dump(curr_motion, f)
            print("")

    # 处理完成后的总结
    print(f"\n处理完成!")
    print(f"总动作数: {max_id}")
    print(f"一直有碰撞动作: {bad_data}")
    print(f"碰撞率: {bad_frame * 100. /(cleaned_frame + bad_frame)}")

if __name__ == "__main__":
    main()