
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from rich import print
# from .params import ROBOT_XML_DICT, IK_CONFIG_DICT

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        ik_config_path: str,
        tgt_robot_xml_path: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = tgt_robot_xml_path
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")

            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        # with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
        with open(ik_config_path) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", ik_config_path)
            # print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        # self.human_scale_table = ik_config["human_scale_table"]
        self.human_scale_table = np.array([ik_config["human_scale_table"][key] for key in ik_config["human_scale_table"].keys()])
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = []
        self.human_body_to_task2 = []
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.qpos_init = self.configuration.data.qpos[3:].copy()
        
        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        rot_offsets1 = []
        pos_offsets1 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1.append(task)
                rot_offsets1.append(rot_offset)
                pos_offsets1.append(np.array(pos_offset) - self.ground)
                self.tasks1.append(task)
                self.task_errors1[task] = []
        self.rot_offsets1 = R.from_quat(rot_offsets1)
        self.pos_offsets1 = np.array(pos_offsets1)
        

        rot_offsets2 = []
        pos_offsets2 = []
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2.append(task)
                rot_offsets2.append(rot_offset)
                pos_offsets2.append(np.array(pos_offset) - self.ground)
                self.tasks2.append(task)
                self.task_errors2[task] = []
        self.rot_offsets2 = R.from_quat(rot_offsets2)
        self.pos_offsets2 = np.array(pos_offsets2)
  
    def update_targets(self, pos, rot, offset_to_ground=False):

        ######## rot:[w, x, y, z]

        pos = self.scale_human_pos(pos, self.human_scale_table)

        if offset_to_ground:
            pos = self.offset_pos_to_ground(pos)

        if self.use_ik_match_table1:
            for i in range(len(self.human_body_to_task1)):
                task = self.human_body_to_task1[i]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot[i]), pos[i]))
        
        if self.use_ik_match_table2:
            for i in range(len(self.human_body_to_task2)):
                task = self.human_body_to_task2[i]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot[i]), pos[i]))
            
    def retarget(self, human_data, offset_to_ground=False):
        
        self.configuration.data.qpos[3:] = self.qpos_init
        self.configuration.data.qvel[3:] = 0.
        mj.mj_forward(self.configuration.model, self.configuration.data)

        pos, rot = human_data[:, :3], human_data[:, 3:]

        # Update the task targets
        self.update_targets(pos, rot, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
        
        xyz = self.configuration.data.xpos[1:].copy()
        rot = self.configuration.data.xquat[1:].copy()
        return np.concatenate([xyz, rot], axis=-1), self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )
    
    def scale_human_pos(self, pos, human_scale_table):
        root_pos = pos[0]
        scaled_root_pos = human_scale_table[0] * root_pos
        human_pos_local = (pos - root_pos) * human_scale_table[:, np.newaxis]
        human_pos_global = human_pos_local + scaled_root_pos
        return human_pos_global
    
    def offset_human_pos_rot(self, pos, rot, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        
        rot = rot[:, [1, 2, 3, 0]] #wxyz -> xyzw
        
        updated_quat = (R.from_quat(rot) * rot_offsets).as_quat() #xyzw

        global_pos_offset = R.from_quat(updated_quat).apply(pos_offsets)
        global_pos = pos + global_pos_offset

        return global_pos, updated_quat

    def offset_pos_to_ground(self, pos):
        ground_offset = 0.05
        lowest_pos = pos[:, 2].min()
        offset_pos = pos.copy()
        offset_pos[:, 2] = offset_pos[:, 2] - lowest_pos + ground_offset

        return offset_pos