


import json
import numpy as np
from scipy.spatial.transform import Rotation as R


UE_links = ['pelvis', # 0
    'left_hip_link', 'left_knee_link', 'left_foot_link', 'left_toe_link', # 1, 2, 3, 4
    'right_hip_link', 'right_knee_link', 'right_foot_link', 'right_toe_link', # 5, 6, 7, 8
    'spine0_link', 'spine1_link', 'spine2_link', # 9, 10, 11
    'left_thorax_link', 'left_shoulder_link', 'left_elbow_link', 'left_wrist_link', # 12, 13, 14, 15
    'right_thorax_link', 'right_shoulder_link', 'right_elbow_link', 'right_wrist_link', # 16, 17, 18, 19
    'neck0_link', 'neck1_link', 'head_link' # 20, 21, 22
]

UE_keypoints_links = ['pelvis', 
    'left_hip_link', 'left_knee_link', 'left_foot_link',
    'right_hip_link', 'right_knee_link', 'right_foot_link',
    'left_shoulder_link', 'left_elbow_link', 'left_wrist_link', 
    'right_shoulder_link', 'right_elbow_link', 'right_wrist_link', 
]

UE_keypoints_parents = {
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

UE_link_parents = {
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

UE_parents_indices = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10,
    11, 12, 13, 14,
    11, 16, 17, 18,
    11, 20, 21
]

g1_links = [
    'pelvis', # 0
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', # 1, 2, 3, 4, 5, 6
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', # 7, 8, 9, 10, 11, 12
    'waist_yaw_link', 'waist_roll_link', 'torso_link', # 13, 14, 15
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', # 16, 17, 18, 19
    'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', # 20, 21, 22
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', # 23, 24, 25, 26
    'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', # 27, 28, 29
]

g1_matched_links = [
    'pelvis',
    'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link',
    'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
    'right_shoulder_roll_link', 'left_elbow_link', 'right_wrist_yaw_link'
]

"""
links = [
    'Hips', # 0
    'RightUpLeg', 'RightLeg', 'RightFoot', # 1, 2, 3 
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', # 4, 5, 6
    'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head', # 7, 8, 9, 10, 11, 12
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', # 13, 14, 15, 16
    'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', # 17, 18, 19
    'RightInHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', # 20, 21, 22, 23 
    'RightInHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', # 24, 25, 26, 27
    'RightInHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', # 28, 29, 30, 31
    'RightInHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', # 32, 33, 34, 35
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', # 36, 37, 38, 39
    'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', # 40, 41, 42
    'LeftInHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', # 43, 44, 45, 46 
    'LeftInHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', # 47, 48, 49, 50
    'LeftInHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', # 51, 52, 53, 54
    'LeftInHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', # 55, 56, 57, 58
]

links_parent = {
    "Hips": "world",
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg",
    "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Neck": "Spine2", "Neck1": "Neck", "Head": "Neck1",
    "RightShoulder": "Spine2", "RightArm": "RightShoulder", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand", "RightHandThumb2": "RightHandThumb1", "RightHandThumb3": "RightHandThumb2",
    "RightInHandIndex": "RightHand", "RightHandIndex1": "RightInHandIndex", "RightHandIndex2": "RightHandIndex1", "RightHandIndex3": "RightHandIndex2",
    "RightInHandMiddle": "RightHand", "RightHandMiddle1": "RightInHandMiddle", "RightHandMiddle2": "RightHandMiddle1", "RightHandMiddle3": "RightHandMiddle2",
    "RightInHandRing": "RightHand", "RightHandRing1": "RightInHandRing", "RightHandRing2": "RightHandRing1", "RightHandRing3": "RightHandRing2",
    "RightInHandPinky": "RightHand", "RightHandPinky1": "RightInHandPinky", "RightHandPinky2": "RightHandPinky1", "RightHandPinky3": "RightHandPinky2",
    "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand", "LeftHandThumb2": "LeftHandThumb1", "LeftHandThumb3": "LeftHandThumb2", 
    "LeftInHandIndex": "LeftHand", "LeftHandIndex1": "LeftInHandIndex", "LeftHandIndex2": "LeftHandIndex1", "LeftHandIndex3": "LeftHandIndex2",
    "LeftInHandMiddle": "LeftHand", "LeftHandMiddle1": "LeftInHandMiddle", "LeftHandMiddle2": "LeftHandMiddle1", "LeftHandMiddle3": "LeftHandMiddle2",
    "LeftInHandRing": "LeftHand", "LeftHandRing1": "LeftInHandRing", "LeftHandRing2": "LeftHandRing1", "LeftHandRing3": "LeftHandRing2",
    "LeftInHandPinky": "LeftHand", "LeftHandPinky1": "LeftInHandPinky", "LeftHandPinky2": "LeftHandPinky1", "LeftHandPinky3": "LeftHandPinky2"
}

# g1_links = [
#     'pelvis', # 0
#     'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', # 1, 2, 3, 4, 5, 6
#     'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', # 7, 8, 9, 10, 11, 12
#     'waist_yaw_link', 'waist_roll_link', 'torso_link', # 13, 14, 15
#     'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', # 16, 17, 18, 19
#     'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', # 20, 21, 22
#     'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', # 23, 24, 25, 26
#     'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', # 27, 28, 29
# ]

g1_links_parent = {
    'pelvis': 'world',
    'left_hip_pitch_link': 'pelvis', 'left_hip_roll_link': 'left_hip_pitch_link', 'left_hip_yaw_link': 'left_hip_roll_link', 'left_knee_link': 'left_hip_yaw_link', 'left_ankle_pitch_link': 'left_knee_link', 'left_ankle_roll_link': 'left_ankle_pitch_link',
    'right_hip_pitch_link': 'pelvis', 'right_hip_roll_link': 'right_hip_pitch_link', 'right_hip_yaw_link': 'right_hip_roll_link', 'right_knee_link': 'right_hip_yaw_link', 'right_ankle_pitch_link': 'right_knee_link', 'right_ankle_roll_link': 'right_ankle_pitch_link',
    'waist_yaw_link': 'pelvis', 'waist_roll_link': 'waist_yaw_link', 'torso_link': 'waist_roll_link',
    'left_shoulder_pitch_link': 'torso_link', 'left_shoulder_roll_link': 'left_shoulder_pitch_link', 'left_shoulder_yaw_link': 'left_shoulder_roll_link', 'left_elbow_link': 'left_shoulder_yaw_link',
    'left_wrist_roll_link': 'left_elbow_link', 'left_wrist_pitch_link': 'left_wrist_roll_link', 'left_wrist_yaw_link': 'left_wrist_pitch_link',
    'right_shoulder_pitch_link': 'torso_link', 'right_shoulder_roll_link': 'right_shoulder_pitch_link', 'right_shoulder_yaw_link': 'right_shoulder_roll_link', 'right_elbow_link': 'right_shoulder_yaw_link',
    'right_wrist_roll_link': 'right_elbow_link', 'right_wrist_pitch_link': 'right_wrist_roll_link', 'right_wrist_yaw_link': 'right_wrist_pitch_link',
}

g1_links_w = [
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link", "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]    

def compute_global_transform(joint, global_transforms, link_poses):
    if joint in global_transforms:
        return global_transforms[joint]

    if joint == "world":
        # World frame is the origin
        global_transforms[joint] = (np.array([0, 0, 0]), R.from_quat([0, 0, 0, 1]), R.from_quat([0, 0, 0, 1]))
        return global_transforms[joint]

    parent = links_parent[joint]
    local_pos, local_quat = np.array(link_poses[joint][:3]), np.array(link_poses[joint][3:])
    # local_quat = np.array([0., 0., 0., 1.])
    parent_global_pos, parent_global_rot, new_parent_global_rot = compute_global_transform(parent, global_transforms, link_poses)

    # Compute global position and orientation
    global_pos = parent_global_pos + parent_global_rot.apply(local_pos)
    global_rot = parent_global_rot * R.from_quat(local_quat)
    new_global_rot = new_parent_global_rot * R.from_quat(local_quat)

    new_global_rot = global_rot

    if joint == "LeftArm":
        original_quat = global_rot.as_quat()
        adjust = R.from_euler('x', 90, degrees=True)
        new_global_rot = global_rot * adjust
    elif joint == "RightArm":
        adjust = R.from_euler('x', -90, degrees=True)
        new_global_rot = global_rot * adjust
    elif joint == "LeftForeArm":
        adjust = R.from_euler('x', 90, degrees=True)
        adjust1 = R.from_euler('y', 90, degrees=True)
        new_global_rot = global_rot * adjust * adjust1
    elif joint == "RightForeArm":
        adjust = R.from_euler('x', -90, degrees=True)
        adjust1 = R.from_euler('y', 90, degrees=True)
        new_global_rot = global_rot * adjust * adjust1
    elif joint == "LeftHand":
        adjust = R.from_euler('x', 90, degrees=True)
        adjust1 = R.from_euler('y', 90, degrees=True)
        new_global_rot = global_rot * adjust * adjust1
    elif joint == "RightHand":
        adjust = R.from_euler('x', -90, degrees=True)
        adjust1 = R.from_euler('y', 90, degrees=True)
        new_global_rot = global_rot * adjust * adjust1

    global_transforms[joint] = (global_pos, global_rot, new_global_rot)

    return global_transforms[joint]

# Define skeletal connections (pairs of joints that should be connected)
skeleton_connections = [
    # Spine and head
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Head"),
    
    # Left arm
    ("Spine2", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),

    # # Left hand fingers
    # ("LeftHand", "LeftHandThumb1"),
    # ("LeftHandThumb1", "LeftHandThumb2"),
    # ("LeftHandThumb2", "LeftHandThumb3"),
    # ("LeftHand", "LeftInHandIndex"),
    # ("LeftInHandIndex", "LeftHandIndex1"),
    # ("LeftHandIndex1", "LeftHandIndex2"),
    # ("LeftHandIndex2", "LeftHandIndex3"),
    # ("LeftHand", "LeftInHandMiddle"),
    # ("LeftInHandMiddle", "LeftHandMiddle1"),
    # ("LeftHandMiddle1", "LeftHandMiddle2"),
    # ("LeftHandMiddle2", "LeftHandMiddle3"),
    # ("LeftHand", "LeftInHandRing"),
    # ("LeftInHandRing", "LeftHandRing1"),
    # ("LeftHandRing1", "LeftHandRing2"),
    # ("LeftHandRing2", "LeftHandRing3"),
    # ("LeftHand", "LeftInHandPinky"),
    # ("LeftInHandPinky", "LeftHandPinky1"),
    # ("LeftHandPinky1", "LeftHandPinky2"),
    # ("LeftHandPinky2", "LeftHandPinky3"),
    
    # Right arm
    ("Spine2", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),

    # # Right hand fingers
    # ("RightHand", "RightHandThumb1"),
    # ("RightHandThumb1", "RightHandThumb2"),
    # ("RightHandThumb2", "RightHandThumb3"),
    # ("RightHand", "RightInHandIndex"),
    # ("RightInHandIndex", "RightHandIndex1"),
    # ("RightHandIndex1", "RightHandIndex2"),
    # ("RightHandIndex2", "RightHandIndex3"),
    # ("RightHand", "RightInHandMiddle"),
    # ("RightInHandMiddle", "RightHandMiddle1"),
    # ("RightHandMiddle1", "RightHandMiddle2"),
    # ("RightHandMiddle2", "RightHandMiddle3"),
    # ("RightHand", "RightInHandRing"),
    # ("RightInHandRing", "RightHandRing1"),
    # ("RightHandRing1", "RightHandRing2"),
    # ("RightHandRing2", "RightHandRing3"),
    # ("RightHand", "RightInHandPinky"),
    # ("RightInHandPinky", "RightHandPinky1"),
    # ("RightHandPinky1", "RightHandPinky2"),
    # ("RightHandPinky2", "RightHandPinky3"),
    
    # Left leg
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    
    # Right leg
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),

]
"""
