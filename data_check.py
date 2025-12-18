import os
import glob
import joblib
from copy import deepcopy

def check_data(data_dir: str):

    all_files = glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)

    target_keys = ["body_pos", "body_rot", "body_lin_vel", "body_ang_vel", "reset_root_rot", "reset_root_trans", "reset_joint_pos", "reset_joint_vel", "robot"]

    for file in all_files:
        motion_data = joblib.load(file)

        motion_keys = motion_data.keys()

        print(motion_keys)

        for target_key in target_keys:

            if target_key not in motion_keys:
                
                raise KeyError(f"Missing Motion key '{motion_key}' in file: {file}")

        for motion_key in motion_keys:

            # print(motion_data[motion_key])

            if motion_key not in target_keys: 
                raise KeyError(f"Unexpected Motion key '{motion_key}' in file: {file}")

            if motion_data[motion_key] is None or len(motion_data[motion_key]) == 0:
                raise ValueError(f"{file}: key '{motion_key}' has no data in file {file}")

def rename_data(src_dir: str, target_dir: str):

    os.makedirs(target_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(src_dir, "*.pkl"), recursive=True)

    for file in all_files:

        motion = joblib.load(file)
        new_motion = deepcopy(motion)
        motion_key = os.path.basename(file).replace(".pkl", "")
        
        new_motion["reset_joint_pos"] = new_motion.pop("joint_pos")
        new_motion["reset_joint_vel"] = new_motion.pop("joint_vel")
        new_motion["reset_root_trans"] = new_motion.pop("relabel_root_trans")
        new_motion["reset_root_rot"] = new_motion.pop("relabel_root_rot")

        assert len(new_motion["body_pos"].shape) == 3
        assert len(new_motion["body_rot"].shape) == 3
        assert len(new_motion["body_lin_vel"].shape) == 3
        assert len(new_motion["body_ang_vel"].shape) == 3
        assert len(new_motion["reset_root_trans"].shape) == 2
        assert len(new_motion["reset_root_rot"].shape) == 2
        assert len(new_motion["reset_joint_pos"].shape) == 2
        assert len(new_motion["reset_joint_vel"].shape) == 2
        
        joblib.dump(new_motion, os.path.join(target_dir, f"{motion_key}.pkl"))
        print(f"data save to: {target_dir}")

data_dir = "Datasets/target_data/g1_29dof/X/251217"
target_dir = "Datasets/target_data/g1_29dof/X_RRRRRRRRR/251217"
rename_data(data_dir, target_dir)
check_data(target_dir)