try:
    import isaacgym
    from isaacgym import gymapi, gymtorch
except ImportError as e:
    from rich import print

    print(
        """[red][Isaac Gym Import Error]
  1. You need to install Isaac Gym, if not installed.
    - Download Isaac Gym following https://clvrai.github.io/furniture-bench/docs/getting_started/installation_guide_furniture_sim.html#download-isaac-gym
    - Then, pip install -e isaacgym/python
  2. If PyTorch was imported before furniture_bench, please import torch after furniture_bench.[/red]
"""
    )
    print()
    raise ImportError(e)

from typing import Union
from datetime import datetime
from pathlib import Path

import torch
import cv2
import gym
import numpy as np

import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.envs.initialization_mode import Randomness, str_to_enum, InitializationMode, str_to_init_mode
from furniture_bench.controllers.osc import osc_factory
from furniture_bench.furniture import furniture_factory
from furniture_bench.sim_config import sim_config
from furniture_bench.assemble_config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import get_mat, rot_mat, rot_mat_to_angles

from furniture_bench.furniture.parts.part import Part

from typing import Any, Dict

ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets")



class FurnitureSimEnv(gym.Env):
    """FurnitureSim base class."""

    def __init__(
            self,
            furniture: str,
            num_envs: int = 1,
            resize_img: bool = True,
            headless: bool = False,
            compute_device_id: int = 0,
            graphics_device_id: int = 0,
            np_step_out: bool = False,
            channel_first: bool = False,
            save_camera_input: bool = False,
            record: bool = False,
            max_env_steps: int = 3000,
            act_rot_repr: str = "quat",

            init_mode: Union[str, InitializationMode] = "pre_assembled",
            set_friction: bool = True,
            task_config: Dict[str, Any] = None,
            furniture_scale_factor=None,

            **kwargs,
    ):
        """
        Args:
            furniture (str): Specifies the type of furniture. Options are 'lamp', 'square_table', 'desk', 'drawer', 'cabinet', 'round_table', 'stool', 'chair', 'one_leg'.
            num_envs (int): Number of parallel environments.
            resize_img (bool): If true, images are resized to 224 x 224.
            obs_keys (list): List of observations for observation space (i.e., RGB-D image from three cameras, proprioceptive states, and poses of the furniture parts.)
            headless (bool): If true, simulation runs without GUI.
            compute_device_id (int): GPU device ID used for simulation.
            graphics_device_id (int): GPU device ID used for rendering.
            np_step_out (bool): If true, env.step() returns Numpy arrays.
            channel_first (bool): If true, color images are returned in channel first format [3, H, w].
            save_camera_input (bool): If true, the initial camera inputs are saved.
            record (bool): If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps (int): Maximum number of steps per episode (default: 3000).
            act_rot_repr (str): Representation of rotation for action space. Options are 'quat', 'axis', or 'rot_6d'.
        """
        super(FurnitureSimEnv, self).__init__()

        self.furniture_name = furniture
        self.num_envs = num_envs
        self.task_config = task_config
        if "box_config" in self.task_config:
            self.box_config = self.task_config["box_config"]
        else:
            self.box_config = None

        self.parts_name = self.task_config["part_names"]

        self.init_mode = str_to_init_mode(init_mode)
        self.set_friction = set_friction

        self.np_step_out = np_step_out
        self.channel_first = channel_first

        self.headless = headless
        self.move_neutral = False
        self.ctrl_started = False

        if furniture_scale_factor is not None:
            self.furniture_scale_factor = furniture_scale_factor
        if "scale_factor" in self.task_config and self.task_config["scale_factor"] is not None:
            self.furniture_scale_factor = self.task_config["scale_factor"]
        else:
            self.furniture_scale_factor = 1.0

        self.reset_check = False
        self.last_check_step = 0
        self.moved = []
        self.sampled_points = []
        self.sampled_normals = []
        self.action_directs = []

        self.device = torch.device("cuda", compute_device_id) if sim_config[
            "sim_params"].use_gpu_pipeline else torch.device("cpu")

        self.furnitures = [furniture_factory(furniture) for _ in range(num_envs)]
        if num_envs == 1:
            self.furniture = self.furnitures[0]
        else:
            self.furniture = furniture_factory(furniture)
        self.furniture.max_env_steps = max_env_steps
        for furn in self.furnitures:
            furn.max_env_steps = max_env_steps

        self.hand_close = torch.tensor([True] * num_envs, device=self.device)

        self.pose_dim = 7
        self.resize_img = resize_img

        self.save_camera_input = save_camera_input
        self.last_grasp = torch.tensor([-1.0] * num_envs, device=self.device)

        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open an close actions.
        self.max_gripper_width = config["robot"]["max_gripper_width"][furniture]
        self.gripper_pos_control = kwargs.get("gripper_pos_control", False)

        self.img_size = sim_config["camera"][
            "resized_img_size" if resize_img else "color_img_size"
        ]

        # Simulator setup.
        self.isaac_gym = gymapi.acquire_gym()

        self.sim = self.isaac_gym.create_sim(
            compute_device_id,
            graphics_device_id,
            gymapi.SimType.SIM_PHYSX,
            sim_config["sim_params"],
        )

        self._create_ground_plane()
        self._setup_lights()

        self.import_assets()
        self.create_envs()
        self.viewer = None
        self.set_viewer()
        self.set_camera()
        self.acquire_base_tensors()

        self.isaac_gym.prepare_sim(self.sim)
        self.refresh()

        self.isaac_gym.refresh_actor_root_state_tensor(self.sim)

        # self.init_ee_pos, self.init_ee_quat = self.get_ee_pose()

        gym.logger.set_level(gym.logger.INFO)

        self.record = record
        if self.record:
            record_dir = Path("sim_record") / datetime.now().strftime("%Y%m%d-%H%M%S")
            record_dir.mkdir(parents=True, exist_ok=True)
            self.video_writer1 = cv2.VideoWriter(
                str(record_dir / "front_video.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                60,
                (self.img_size[0], self.img_size[1]),
            )
            self.video_writer2 = cv2.VideoWriter(
                str(record_dir / "back_video.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                60,
                (self.img_size[0], self.img_size[1]),
            )
            self.video_writer3 = cv2.VideoWriter(
                str(record_dir / "left_video.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                60,
                (self.img_size[0], self.img_size[1]),
            )
            self.video_writer4 = cv2.VideoWriter(
                str(record_dir / "right_video.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                60,
                (self.img_size[0], self.img_size[1]),
            )

        if act_rot_repr != "quat" and act_rot_repr != "axis" and act_rot_repr != "rot_6d":
            raise ValueError(f"Invalid rotation representation: {act_rot_repr}")
        self.act_rot_repr = act_rot_repr

        self.robot_state_as_dict = kwargs.get("robot_state_as_dict", True)
        self.squeeze_batch_dim = kwargs.get("squeeze_batch_dim", False)

    def _create_ground_plane(self):
        """Creates ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.isaac_gym.add_ground(self.sim, plane_params)

    def _setup_lights(self):
        for light in sim_config["lights"]:
            l_color = gymapi.Vec3(*light["color"])
            l_ambient = gymapi.Vec3(*light["ambient"])
            l_direction = gymapi.Vec3(*light["direction"])
            self.isaac_gym.set_light_parameters(
                self.sim, 0, l_color, l_ambient, l_direction
            )

    def create_envs(self):
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        table_half_width = 0.015
        table_surface_z = table_pos.z + table_half_width

        self.franka_pose = gymapi.Transform()
        self.franka_pose.p = gymapi.Vec3(
            0.5 * -table_pos.x + 0.1, 0.35, table_surface_z + ROBOT_HEIGHT
        )

        if "position" in config["robot"].keys():
            self.franka_pose.p = gymapi.Vec3(
                config["robot"]["position"][0],
                config["robot"]["position"][1],
                config["robot"]["position"][2],
            )

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p.x, self.franka_pose.p.y, self.franka_pose.p.z],
            [0, 0, 0],
        )
        self.table_from_origin_mat = get_mat(
            [0, 0, table_surface_z],
            [0, 0, 0],
        )
        
        franka_link_dict = self.isaac_gym.get_asset_rigid_body_dict(self.franka_asset)
        franka_hand_link_dict = self.isaac_gym.get_asset_rigid_body_dict(self.franka_hand_asset)
        self.franka_ee_index = franka_link_dict["k_ee_link"]
        self.franka_hand_index = franka_hand_link_dict["k_ee_link"]
        self.franka_base_index = franka_link_dict["panda_link0"]
        self.franka_hand_base_index = franka_hand_link_dict["panda_hand"]

        # Parts assets.
        # Create assets.
        self.part_assets = {}
        for part in self.furniture.parts:
            if part.name not in self.task_config["part_names"]:
                continue
            asset_option = sim_config["asset"][part.name]
            self.part_assets[part.name] = self.isaac_gym.load_asset(
                self.sim, ASSET_ROOT, part.asset_file, asset_option
            )

        # Create envs.
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.handles = {}

        self.ee_idxs = []
        self.ee_handles = []
        self.osc_ctrls = []
        self.pos_ctrls = []
        self.base_idxs = []
        self.franka_handles = []
        self.part_idxs = {}
        self.part_reset = {}

        self.observation = [[] * self.num_envs]
        self.record_obs = [False * self.num_envs]

        # zhp: changed
        for i in range(self.num_envs):
            env = self.isaac_gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # Add workspace (table).
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)
            self.table_pose = table_pose

            table_handle = self.isaac_gym.create_actor(
                env, self.table_asset, table_pose, "table", i, 0
            )
            self.table_actor_index = self.isaac_gym.find_actor_index(
                env, "table", gymapi.DOMAIN_ENV
            )

            table_props = self.isaac_gym.get_actor_rigid_shape_properties(
                env, table_handle
            )
            table_props[0].friction = sim_config["table"]["friction"]

            self.isaac_gym.set_actor_rigid_shape_properties(
                env, table_handle, table_props
            )

            self.base_table_rigid_index = self.isaac_gym.get_actor_rigid_body_index(
                env, table_handle, 0, gymapi.DOMAIN_SIM
            )

            if self.box_config is not None:
                box_pose = gymapi.Transform()
                if "pos" in self.box_config.keys():
                    box_pose.p = gymapi.Vec3(
                        self.box_config["pos"][0], self.box_config["pos"][1],
                        table_pos.z + table_half_width + self.box_config["size"][2] / 2
                    )
                else:
                    box_pose.p = gymapi.Vec3(
                        0.0, 0.0, table_pos.z + table_half_width + self.box_config["size"][2] / 2
                    )
                box_handle = self.isaac_gym.create_actor(
                    env, self.box_asset, box_pose, "box", i, 0
                )
                box_props = self.isaac_gym.get_actor_rigid_shape_properties(
                    env, box_handle
                )
                box_props[0].friction = 0
                self.isaac_gym.set_actor_rigid_shape_properties(
                    env, box_handle, box_props
                )

            # zhp: add fixed table
            fixed_table_pose = gymapi.Transform()
            fixed_table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z - 0.03)

            fixed_table_handle = self.isaac_gym.create_actor(
                env, self.fixed_table_asset, fixed_table_pose, "fixed_table", i, 0
            )

            fixed_table_props = self.isaac_gym.get_actor_rigid_shape_properties(
                env, fixed_table_handle
            )
            # table_props[0].friction = sim_config["table"]["friction"]
            fixed_table_props[0].friction = 1000
            self.isaac_gym.set_actor_rigid_shape_properties(
                env, fixed_table_handle, fixed_table_props
            )

            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)
            bg_handle = self.isaac_gym.create_actor(
                env, self.background_asset, bg_pose, "background", i, 0
            )

            # Add robot.
            franka_handle = self.isaac_gym.create_actor(
                env, self.franka_asset, self.franka_pose, "franka", i, 0
            )

            self.franka_handle_index = self.isaac_gym.find_actor_index(
                env, "franka", gymapi.DOMAIN_ENV
            )

            self.franka_num_dofs = self.isaac_gym.get_actor_dof_count(
                env, franka_handle
            )

            self.isaac_gym.enable_actor_dof_force_sensors(env, franka_handle)
            self.franka_handles.append(franka_handle)

            # Get global index of hand and base.
            self.ee_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka_handle, self.franka_ee_index, gymapi.DOMAIN_SIM
                )
            )
            self.ee_handles.append(
                self.isaac_gym.find_actor_rigid_body_handle(
                    env, franka_handle, "k_ee_link"
                )
            )
            self.base_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka_handle, self.franka_base_index, gymapi.DOMAIN_SIM
                )
            )

            # Set dof properties.
            franka_dof_props = self.isaac_gym.get_asset_dof_properties(
                self.franka_asset
            )
            # Franka
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][:7].fill(0.0)
            franka_dof_props["damping"][:7].fill(0.0)
            franka_dof_props["friction"][:7] = sim_config["robot"]["arm_frictions"]
            # Grippers
            if self.gripper_pos_control:
                franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
                franka_dof_props["stiffness"][7:].fill(200.0)
                franka_dof_props["damping"][7:].fill(60.0)
            else:
                franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_EFFORT)
                franka_dof_props["stiffness"][7:].fill(0)
                franka_dof_props["damping"][7:].fill(0)
                franka_dof_props["friction"][7:] = sim_config["robot"]["gripper_frictions"]
            franka_dof_props["upper"][7:] = self.max_gripper_width / 2

            self.isaac_gym.set_actor_dof_properties(
                env, franka_handle, franka_dof_props
            )

            # Set initial dof states
            franka_num_dofs = self.isaac_gym.get_asset_dof_count(self.franka_asset)
            self.default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
            self.default_dof_pos[:7] = np.array(
                config["robot"]["reset_joints"], dtype=np.float32
            )
            self.default_dof_pos[7:] = self.max_gripper_width / 2
            default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
            default_dof_state["pos"] = self.default_dof_pos
            self.isaac_gym.set_actor_dof_states(
                env, franka_handle, default_dof_state, gymapi.STATE_ALL
            )

            # zhp: add robot hand
            hand_ori = get_mat([0, 0, 0], [0, np.pi / 2, np.pi / 2])
            hand_pos = [0.0, 0.0, 1.5]

            self.franka_hand_pose = gymapi.Transform()
            self.franka_hand_pose.p = gymapi.Vec3(
                hand_pos[0], hand_pos[1], hand_pos[2]
            )
            self.franka_hand_pose.r = gymapi.Quat(*T.mat2quat(hand_ori[:3, :3]))

            self.franka_hand_handle = self.isaac_gym.create_actor(
                env, self.franka_hand_asset, self.franka_hand_pose, "franka_hand", i, 0
            )
            self.franka_hand_actor_index = self.isaac_gym.find_actor_index(
                env, "franka_hand", gymapi.DOMAIN_ENV
            )
            self.franka_hand_rigid_index = self.isaac_gym.get_actor_rigid_body_index(
                env, self.franka_hand_handle, 0, gymapi.DOMAIN_SIM
            )

            self.isaac_gym.set_rigid_body_segmentation_id(
                env, self.franka_hand_handle, 0, 10
            )

            # Set dof properties.
            franka_hand_dof_props = self.isaac_gym.get_asset_dof_properties(
                self.franka_hand_asset
            )
            # Grippers
            if self.gripper_pos_control:
                franka_hand_dof_props["driveMode"][0:].fill(gymapi.DOF_MODE_POS)
                franka_hand_dof_props["stiffness"][0:].fill(200.0)
                franka_hand_dof_props["damping"][0:].fill(60.0)
            else:
                franka_hand_dof_props["driveMode"][0:].fill(gymapi.DOF_MODE_EFFORT)
                franka_hand_dof_props["stiffness"][0:].fill(0)
                franka_hand_dof_props["damping"][0:].fill(0)
                franka_hand_dof_props["friction"][0:] = sim_config["robot"]["fgripper_frictions"]
            franka_hand_dof_props["upper"][0:] = self.max_gripper_width / 2

            self.isaac_gym.set_actor_dof_properties(
                env, self.franka_hand_handle, franka_hand_dof_props
            )

            # Set initial dof states
            franka_hand_num_dofs = self.isaac_gym.get_asset_dof_count(self.franka_hand_asset)
            self.default_hand_dof_pos = np.zeros(franka_hand_num_dofs, dtype=np.float32)
            self.default_hand_dof_pos[0:] = self.max_gripper_width / 2
            default_hand_dof_state = np.zeros(franka_hand_num_dofs, gymapi.DofState.dtype)
            default_hand_dof_state["pos"] = self.default_hand_dof_pos
            self.isaac_gym.set_actor_dof_states(
                env, franka_handle, default_hand_dof_state, gymapi.STATE_ALL
            )

            # Add furniture parts.
            poses = []
            for part in self.furniture.parts:
                if part.name not in self.task_config["part_names"]:
                    continue
                pos, ori = self._get_reset_pose(part)

                part_pose_mat = self.table_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
                part_pose = gymapi.Transform()
                part_pose.p = gymapi.Vec3(
                    part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
                )
                reset_ori = self.table_coord_to_sim_coord(ori)
                part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
                poses.append(part_pose)
                part_handle = self.isaac_gym.create_actor(
                    env, self.part_assets[part.name], part_pose, part.name, i, 0
                )

                # zhp: scaled
                self.isaac_gym.set_actor_scale(
                    env, part_handle, self.furniture_scale_factor
                )

                self.handles[part.name] = part_handle

                part_idx = self.isaac_gym.get_actor_rigid_body_index(
                    env, part_handle, 0, gymapi.DOMAIN_SIM
                )

                # print(part.name, part_idx)
                # Set properties of part.

                if self.set_friction and "part_frictions" in self.task_config.keys():
                    part_props = self.isaac_gym.get_actor_rigid_shape_properties(
                        env, part_handle
                    )
                    if part.name in self.parts_name:
                        index = self.parts_name.index(part.name)
                        part_props[0].friction = self.task_config["part_frictions"][index]
                    else:
                        raise ValueError(f"Unknown part name: {part.name}")

                    self.isaac_gym.set_actor_rigid_shape_properties(
                        env, part_handle, part_props
                    )

                if "part_mass" in self.task_config.keys():
                    part_body_props = self.isaac_gym.get_actor_rigid_body_properties(
                        env, part_handle
                    )
                    if part.name in self.parts_name:
                        index = self.parts_name.index(part.name)
                        part_body_props[0].mass = self.task_config["part_mass"][index]
                    else:
                        raise ValueError(f"Unknown part name: {part.name}")
                    self.isaac_gym.set_actor_rigid_body_properties(
                        env, part_handle, part_body_props
                    )

                if part.name in self.parts_name:
                    index = self.parts_name.index(part.name)
                    segmentation_id = index + 1
                else:
                    raise ValueError(f"Unknown part name: {part.name}")

                self.isaac_gym.set_rigid_body_segmentation_id(
                    env, part_handle, 0, segmentation_id
                )
                # print("part name: ", part.name)
                if self.part_idxs.get(part.name) is None:
                    self.part_idxs[part.name] = [part_idx]
                else:
                    self.part_idxs[part.name].append(part_idx)

            self.parts_handles = {}
            for part in self.furniture.parts:
                if part.name not in self.task_config["part_names"]:
                    continue
                self.parts_handles[part.name] = self.isaac_gym.find_actor_index(
                    env, part.name, gymapi.DOMAIN_ENV
                )

        # print(f'Getting the separate actor indices for the frankas and the furniture parts (not the handles)')
        self.franka_actor_idx_all = []
        self.part_actor_idx_all = []  # global list of indices, when resetting all parts
        self.part_actor_idx_by_env = {}  # allow to access part indices based on environment indices
        for env_idx in range(self.num_envs):
            self.franka_actor_idx_all.append(
                self.isaac_gym.find_actor_index(self.envs[env_idx], 'franka', gymapi.DOMAIN_SIM))
            self.part_actor_idx_by_env[env_idx] = []
            for part in self.furnitures[env_idx].parts:
                if part.name not in self.task_config["part_names"]:
                    continue
                part_actor_idx = self.isaac_gym.find_actor_index(self.envs[env_idx], part.name, gymapi.DOMAIN_SIM)
                self.part_actor_idx_all.append(part_actor_idx)
                self.part_actor_idx_by_env[env_idx].append(part_actor_idx)

        self.franka_actor_idxs_all_t = torch.tensor(self.franka_actor_idx_all, device=self.device,
                                                    dtype=torch.int32)
        self.part_actor_idxs_all_t = torch.tensor(self.part_actor_idx_all, device=self.device, dtype=torch.int32)

    def _get_reset_pose(self, part: Part):
        """Get the reset pose of the part.

        Args:
            part: The part to get the reset pose.
        """

        if part.name == self.parts_name[0]:
            pos = config["furniture"][self.furniture_name][part.name]["reset_pos"][0]
            ori = config["furniture"][self.furniture_name][part.name]["reset_ori"][0]
            if "randomness" in self.task_config.keys():
                pos_randomness = self.task_config["randomness"]["pos"]
                ori_randomness = self.task_config["randomness"]["ori"]

                rand_pos = [pos[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                            enumerate(pos_randomness)]
                ori_angles = rot_mat_to_angles(ori)
                rand_ori_angles = [ori_angles[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                   enumerate(ori_randomness)]
                rand_ori = rot_mat(rand_ori_angles, hom=True)

                pos = rand_pos
                ori = rand_ori
        elif part.name == self.parts_name[1]:
            if self.init_mode == InitializationMode.ASSEMBLED:
                attached_part = False
                attach_to = None
                for assemble_pair in self.furniture.should_be_assembled:
                    if part.part_idx == assemble_pair[1]:
                        attached_part = True
                        attach_to = self.furniture.parts[assemble_pair[0]]
                        break
                if attached_part:
                    attach_to_part_name = attach_to.name
                    if attach_to_part_name not in self.parts_name:
                        raise ValueError(
                            f"Part {part.name} should be attached to {attach_to_part_name}, but {attach_to_part_name} is not in the parts list.")
                    if attach_to_part_name not in self.part_reset.keys():
                        raise ValueError(f"Part {attach_to_part_name} should be before {part.name}.")

                    attach_part_pos, attach_part_ori = self.part_reset[attach_to_part_name]
                    attach_part_pose = get_mat(attach_part_pos, attach_part_ori)
                    default_assembled_pose = config["furniture"][self.furniture_name][part.name][
                        "default_assembled_pose"]
                    scaled_default_assembled_pose = default_assembled_pose.copy()
                    scaled_default_assembled_pose[:3, 3] *= self.furniture_scale_factor
                    part_pose = (
                            attach_part_pose
                            @ scaled_default_assembled_pose
                    )
                    pos = part_pose[:3, 3]
                    ori = T.to_hom_ori(part_pose[:3, :3])
                else:
                    pos = config["furniture"][self.furniture_name][part.name]["reset_pos"][0]
                    ori = config["furniture"][self.furniture_name][part.name]["reset_ori"][0]
                    if "randomness" in self.task_config.keys():
                        pos_randomness = self.task_config["randomness"]["pos"]
                        ori_randomness = self.task_config["randomness"]["ori"]

                        rand_pos = [pos[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                    enumerate(pos_randomness)]
                        ori_angles = rot_mat_to_angles(ori)
                        rand_ori_angles = [ori_angles[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                           enumerate(ori_randomness)]
                        rand_ori = rot_mat(rand_ori_angles, hom=True)

                        pos = rand_pos
                        ori = rand_ori
            elif self.init_mode == InitializationMode.PREASSEMBLED:
                scaled_default_assembled_pose = np.load(
                    self.task_config["disassembled_pose_path"],
                    allow_pickle=True,
                )
                scaled_default_assembled_pose[:3, 3] *= self.furniture_scale_factor / 1.5
                # scaled_default_assembled_pose[3, 3] += 0.01
                
                attached_part = False
                attach_to = None
                for assemble_pair in self.furniture.should_be_assembled:
                    if part.part_idx == assemble_pair[1]:
                        attached_part = True
                        attach_to = self.furniture.parts[assemble_pair[0]]
                        break
                if not attached_part:
                    raise ValueError(
                        f"The second part {part.name} should be attached to another part for pre-assembled mode.")
                else:
                    attach_to_part_name = attach_to.name
                    if attach_to_part_name not in self.parts_name:
                        raise ValueError(
                            f"Part {part.name} should be attached to {attach_to_part_name}, but {attach_to_part_name} is not in the parts list.")
                    if attach_to_part_name not in self.part_reset.keys():
                        raise ValueError(f"Part {attach_to_part_name} should be before {part.name}.")
                part1_pose = get_mat(*self.part_reset[attach_to_part_name])
                part2_pose = (
                        part1_pose
                        @ scaled_default_assembled_pose
                )
                pos = part2_pose[:3, 3]
                ori = T.to_hom_ori(part2_pose[:3, :3])
            else:
                pos = config["furniture"][self.furniture_name][part.name]["reset_pos"][0]
                pos[2] = pos[2] * self.furniture_scale_factor + 0.005
                ori = config["furniture"][self.furniture_name][part.name]["reset_ori"][0]
                if "randomness_part" in self.task_config.keys():
                    pos_randomness = self.task_config["randomness_part"]["pos"]
                    ori_randomness = self.task_config["randomness_part"]["ori"]

                    rand_pos = [pos[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                enumerate(pos_randomness)]
                    ori_angles = rot_mat_to_angles(ori)
                    rand_ori_angles = [ori_angles[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                       enumerate(ori_randomness)]
                    rand_ori = rot_mat(rand_ori_angles, hom=True)
                    pos = rand_pos
                    ori = rand_ori
        elif part.name in self.parts_name:
            if self.init_mode == InitializationMode.DISASSEMBLED:
                pos = config["furniture"][self.furniture_name][part.name]["reset_pos"][0]
                pos[2] = pos[2] * self.furniture_scale_factor + 0.005
                ori = config["furniture"][self.furniture_name][part.name]["reset_ori"][0]
                if "randomness_part" in self.task_config.keys():
                    pos_randomness = self.task_config["randomness_part"]["pos"]
                    ori_randomness = self.task_config["randomness_part"]["ori"]

                    rand_pos = [pos[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                enumerate(pos_randomness)]
                    ori_angles = rot_mat_to_angles(ori)
                    rand_ori_angles = [ori_angles[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                       enumerate(ori_randomness)]
                    rand_ori = rot_mat(rand_ori_angles, hom=True)

                    pos = rand_pos
                    ori = rand_ori
            else:
                attached_part = False
                attach_to = None
                for assemble_pair in self.furniture.should_be_assembled:
                    if part.part_idx == assemble_pair[1]:
                        attached_part = True
                        attach_to = self.furniture.parts[assemble_pair[0]]
                        break
                if attached_part:
                    attach_to_part_name = attach_to.name
                    if attach_to_part_name not in self.parts_name:
                        raise ValueError(
                            f"Part {part.name} should be attached to {attach_to_part_name}, but {attach_to_part_name} is not in the parts list.")
                    if attach_to_part_name not in self.part_reset.keys():
                        raise ValueError(f"Part {attach_to_part_name} should be before {part.name}.")

                    attach_part_pos, attach_part_ori = self.part_reset[attach_to_part_name]
                    attach_part_pose = get_mat(attach_part_pos, attach_part_ori)
                    default_assembled_pose = config["furniture"][self.furniture_name][part.name][
                        "default_assembled_pose"]
                    scaled_default_assembled_pose = default_assembled_pose.copy()
                    scaled_default_assembled_pose[:3, 3] *= self.furniture_scale_factor
                    part_pose = (
                        attach_part_pose
                        @ scaled_default_assembled_pose
                    )
                    pos = part_pose[:3, 3]
                    ori = T.to_hom_ori(part_pose[:3, :3])
                else:
                    pos = config["furniture"][self.furniture_name][part.name]["reset_pos"][0]
                    ori = config["furniture"][self.furniture_name][part.name]["reset_ori"][0]
                    if "randomness" in self.task_config.keys():
                        pos_randomness = self.task_config["randomness"]["pos"]
                        ori_randomness = self.task_config["randomness"]["ori"]

                        rand_pos = [pos[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                    enumerate(pos_randomness)]
                        ori_angles = rot_mat_to_angles(ori)
                        rand_ori_angles = [ori_angles[i] + float(np.random.uniform(low, high)) for i, (low, high) in
                                           enumerate(ori_randomness)]
                        rand_ori = rot_mat(rand_ori_angles, hom=True)

                        pos = rand_pos
                        ori = rand_ori
        else:
            raise ValueError("Unknown part name")

        self.part_reset[part.name] = (pos, ori)
        return pos, ori

    def set_viewer(self):
        """Create the viewer."""
        self.enable_viewer_sync = True
        self.viewer = None

        if not self.headless:
            self.viewer = self.isaac_gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )
            # Point camera at middle env.
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.62)
            middle_env = self.envs[0]
            self.isaac_gym.viewer_camera_look_at(
                self.viewer, middle_env, cam_pos, cam_target
            )

    def set_camera(self):
        self.camera_handles = {}
        self.camera_matix = []

        def create_camera(name, i):
            env = self.envs[i]
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = not self.handles
            camera_cfg.width = self.img_size[0]
            camera_cfg.height = self.img_size[1]
            camera_cfg.near_plane = 0.001
            camera_cfg.far_plane = 2.0
            camera_cfg.horizontal_fov = 40.0 if self.resize_img else 69.4
            self.camera_cfg = camera_cfg
            camera_hori_legth = 2.0

            if name == "front":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                cam_pos = gymapi.Vec3(1.0, -0.00, 0.67)
                cam_target = gymapi.Vec3(1.0 - camera_hori_legth, -0.00, 0.32)
                self.isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
                self.front_cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
                self.front_cam_target = np.array(
                    [cam_target.x, cam_target.y, cam_target.z]
                )
            elif name == "back":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                cam_pos = gymapi.Vec3(-0.4, -0.00, 0.67)
                cam_target = gymapi.Vec3(-0.4 + camera_hori_legth, -0.00, 0.32)
                self.isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)

            elif name == "left":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                cam_pos = gymapi.Vec3(0.25, -0.70, 0.67)
                cam_target = gymapi.Vec3(0.25, -0.7 + camera_hori_legth, 0.32)
                self.isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)

            elif name == "right":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                cam_pos = gymapi.Vec3(0.25, 0.70, 0.67)
                cam_target = gymapi.Vec3(0.25, 0.7 - camera_hori_legth, 0.32)
                self.isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)

            return camera

        camera_names = ["front", "back", "left", "right"]

        for env_idx, env in enumerate(self.envs):
            self.camera_matix.append({})
            for camera_name in camera_names:
                if camera_name not in self.camera_handles:
                    self.camera_handles[camera_name] = []
                    self.camera_matix[env_idx][camera_name] = {}
                if len(self.camera_handles[camera_name]) <= env_idx:
                    self.camera_handles[camera_name].append(create_camera(camera_name, env_idx))
                    self.camera_matix[env_idx][camera_name]["proj"] = self.isaac_gym.get_camera_proj_matrix(self.sim,
                                                                                                            self.envs[
                                                                                                                env_idx],
                                                                                                            self.camera_handles[
                                                                                                                camera_name][
                                                                                                                env_idx])
                    self.camera_matix[env_idx][camera_name]["view"] = self.isaac_gym.get_camera_view_matrix(self.sim,
                                                                                                            self.envs[
                                                                                                                env_idx],
                                                                                                            self.camera_handles[
                                                                                                                camera_name][
                                                                                                                env_idx])

    def set_obs(self, env_idx):
        self.record_obs[env_idx] = True

    def reset_obs(self, env_idx):
        self.observation[env_idx] = []

    def _reset_check(self):
        self.reset_check = False
        self.last_check_step = 0
        self.moved = []
        self.sampled_points = []
        self.sampled_normals = []
        self.action_directs = []

    def reset_check_f(self):
        self._reset_check()

    def set_check(self, b: bool = True):
        self.reset_check = b

    def set_last_check_step(self, step):
        self.last_check_step = step

    def get_obs(self, env_idx):
        depth_image = {}
        segments = {}
        pose = {}

        depth_render_type = gymapi.IMAGE_DEPTH
        seg_render_type = gymapi.IMAGE_SEGMENTATION
        for i, handles in self.camera_handles.items():
            # points_, normals_, segments = capture_pc(self,handles[env_idx])

            _depth_image = torch.tensor(
                self.isaac_gym.get_camera_image(
                    self.sim, self.envs[env_idx], handles[env_idx], depth_render_type
                )
            ).to(self.device)
            depth_image[i] = _depth_image.clone()
            seg_image = torch.tensor(
                self.isaac_gym.get_camera_image(
                    self.sim, self.envs[env_idx], handles[env_idx], seg_render_type
                )
            )
            _segments = seg_image.flatten()
            segments[i] = _segments.clone()
        for k, v in self.part_idxs.items():
            pose[k] = self.rb_states[v][env_idx][:7].clone()

        obs = {}
        obs["depth_img"] = depth_image
        obs["segments"] = segments
        obs["part_pose"] = pose
        obs["hand_pose"] = self.rb_states[[self.franka_hand_rigid_index]][env_idx][:7].clone()

        return obs

    def check_contact(self):
        contact_flag = False
        for i in range(10):
            self.refresh()
            self.isaac_gym.refresh_net_contact_force_tensor(self.sim)
            _net_cf = self.isaac_gym.acquire_net_contact_force_tensor(self.sim)
            net_cf = gymtorch.wrap_tensor(_net_cf)
            part1_top_cf = net_cf[self.part_idxs[self.part1_name]].squeeze(0)
            hand_cf = net_cf[self.franka_hand_rigid_index].squeeze(0)
            base_table_cf = net_cf[self.base_table_rigid_index].squeeze(0)

            if torch.any(torch.abs(base_table_cf[:2]) > 50):
                contact_flag = True
                print("Base Table Contact")
                break

            if torch.any(torch.abs(part1_top_cf[:2]) > 100):
                contact_flag = True
                print("Part1 Contact")
                print(part1_top_cf[:2])
                break

        return contact_flag

    def import_assets(self):
        self.background_asset = self._import_background_asset()
        self.table_asset = self._import_table_asset()
        self.fixed_table_asset = self._import_fixed_table_asset()
        self.franka_asset = self._import_franka_asset()
        self.franka_hand_asset = self._import_franka_hand_asset()
        self.box_asset = self._import_box_asset()

    def acquire_base_tensors(self):
        # Get rigid body state tensor
        _rb_states = self.isaac_gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _root_tensor = self.isaac_gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)
        self.root_pos = self.root_tensor.view(self.num_envs, -1, 13)[:self.num_envs, ..., 0:3]
        self.root_quat = self.root_tensor.view(self.num_envs, -1, 13)[:self.num_envs, ..., 3:7]

        self.franka_dof_index = self.isaac_gym.get_actor_dof_index(self.envs[0], self.franka_handles[0], 0,
                                                                   gymapi.DOMAIN_ENV)
        self.franka_hand_dof_index = self.isaac_gym.get_actor_dof_index(self.envs[0], self.franka_hand_handle, 0,
                                                                        gymapi.DOMAIN_ENV)

        _forces = self.isaac_gym.acquire_dof_force_tensor(self.sim)
        _forces = gymtorch.wrap_tensor(_forces)
        # zhp: changed
        self.forces = _forces.view(self.num_envs, -1)[:self.num_envs, ...]

        # Get DoF tensor
        _dof_states = self.isaac_gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(
            _dof_states
        )  # (num_dofs, 2), 2 for pos and vel.
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1)[:self.num_envs, ...]
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1)[:self.num_envs, ...]
        # Get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.isaac_gym.acquire_jacobian_tensor(self.sim, "franka")

        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.jacobian_eef = self.jacobian[
                            :, self.franka_ee_index - 1, :, :7
                            ]  # -1 due to finxed base link.

        # Prepare mass matrix tensor
        # For franka, tensor shape is (num_envs, 7 + 2, 7 + 2), 2 for grippers.
        _massmatrix = self.isaac_gym.acquire_mass_matrix_tensor(self.sim, "franka")

        self.mm = gymtorch.wrap_tensor(_massmatrix)

    # @property
    # def sim_to_robot_mat(self):
    #     return torch.tensor(self.franka_from_origin_mat, device=self.device)

    @property
    def robot_to_ee_mat(self):
        return torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=self.device)
    
    def table_coord_to_sim_coord(self, table_coord_mat):
        """Convert table coordinates to simulation coordinates."""
        return self.table_from_origin_mat @ table_coord_mat
    @property
    def action_space(self):
        # Action space to be -1.0 to 1.0.
        if self.act_rot_repr == "quat":
            pose_dim = 7
        elif self.act_rot_repr == "rot_6d":
            pose_dim = 9
        else:  # axis
            pose_dim = 6

        # zhp : changed
        low = np.array([-1] * pose_dim + [-1], dtype=np.float32)
        high = np.array([1] * pose_dim + [1], dtype=np.float32)
        # low = np.array([-1] * pose_dim + [-1] + [-1] * pose_dim + [-1], dtype=np.float32)
        # high = np.array([1] * pose_dim + [1] + [1] * pose_dim + [1], dtype=np.float32)

        low = np.tile(low, (self.num_envs, 1))
        high = np.tile(high, (self.num_envs, 1))

        return gym.spaces.Box(low, high, (self.num_envs, (pose_dim + 1)))

    @property
    def action_dimension(self):
        return self.action_space.shape[-1]

    @torch.no_grad()
    def step(self, action):
        """Robot takes an action.

        Args:
            action:
                (num_envs, 8): End-effector delta in [x, y, z, qx, qy, qz, qw, gripper] if self.act_rot_repr == "quat".
                (num_envs, 10): End-effector delta in [x, y, z, 6D rotation, gripper] if self.act_rot_repr == "rot_6d".
                (num_envs, 7): End-effector delta in [x, y, z, ax, ay, az, gripper] if self.act_rot_repr == "axis".
        """

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(device=self.device)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        # Clip the action to be within the action space.
        low = torch.from_numpy(self.action_space.low).to(device=self.device)
        high = torch.from_numpy(self.action_space.high).to(device=self.device)
        action = torch.clamp(action, low, high)

        sim_steps = int(
            1.0
            / config["robot"]["hz"]
            / sim_config["sim_params"].dt
            / sim_config["sim_params"].substeps
            + 0.1
        )
        if not self.ctrl_started:
            self.init_ctrl()
        # Set the goal
        ee_pos, ee_quat = self.get_ee_pose_robot()

        for env_idx in range(self.num_envs):
            if self.act_rot_repr == "quat":
                action_quat = action[env_idx][3:7]

            elif self.act_rot_repr == "rot_6d":
                import pytorch3d.transforms as pt
                # Create "actions" dataset.
                rot_6d = action[:, 3:9]
                rot_mat = pt.rotation_6d_to_matrix(rot_6d)
                quat = pt.matrix_to_quaternion(rot_mat)
                action_quat = quat[env_idx]

            else:
                action_quat = C.axisangle2quat(action[env_idx][3:6])

            self.osc_ctrls[env_idx].set_goal(
                action[env_idx][:3] + ee_pos[env_idx],
                C.quat_multiply(ee_quat[env_idx], action_quat).to(self.device),
            )

        for _ in range(sim_steps):
            self.refresh()

            pos_action = torch.zeros_like(self.dof_pos)
            torque_action = torch.zeros_like(self.dof_pos)

            # zhp: changed
            grip_action = torch.zeros((self.num_envs, 2))

            for env_idx in range(self.num_envs):
                grasp = action[env_idx, -1]
                if (
                        torch.sign(grasp) != torch.sign(self.last_grasp[env_idx])
                        and torch.abs(grasp) > self.grasp_margin
                ):
                    grip_sep = self.max_gripper_width if grasp < 0 else 0.0
                    self.last_grasp[env_idx] = grasp
                else:
                    # Keep the gripper open if the grasp has not changed
                    if self.last_grasp[env_idx] < 0:
                        grip_sep = self.max_gripper_width
                    else:
                        grip_sep = 0.0

                grip_action[env_idx, 0] = grip_sep

                state_dict = {}

                ee_pos, ee_quat = self.get_ee_pose_robot()

                state_dict["ee_pose"] = C.pose2mat(
                    ee_pos[env_idx], ee_quat[env_idx], self.device
                ).t()  # OSC expect column major
                state_dict["joint_positions"] = self.dof_pos[env_idx][self.franka_dof_index:self.franka_dof_index + 7]
                state_dict["joint_velocities"] = self.dof_vel[env_idx][
                                                    self.franka_dof_index:self.franka_dof_index + 7]
                state_dict["mass_matrix"] = self.mm[env_idx][
                                            :7, :7
                                            ].t()  # OSC expect column major
                state_dict["jacobian"] = self.jacobian_eef[
                    env_idx
                ].t()  # OSC expect column major
                torques = self.osc_ctrls[env_idx](state_dict)[
                    "joint_torques"
                ]
                torque_action[env_idx, self.franka_dof_index:self.franka_dof_index + 7] = torques
                # zhp: change
                if self.gripper_pos_control:
                    grip_action[env_idx, -1] = grip_sep
                else:
                    if grip_sep > 0:
                        torque_action[env_idx, self.franka_dof_index + 7:self.franka_dof_index + 9] = \
                            sim_config["robot"]["gripper_torque"]
                    else:
                        torque_action[env_idx, self.franka_dof_index + 7:self.franka_dof_index + 9] = - \
                            sim_config["robot"]["gripper_torque"]

            # Gripper action
            if self.gripper_pos_control:
                pos_action[:, self.franka_dof_index + 7:self.franka_dof_index + 9] = grip_action[env_idx, 0]
                pos_action[:,self.franka_hand_dof_index:self.franka_hand_dof_index + 2] = self.max_gripper_width if not self.hand_close else 0.0
            else:
                if not self.hand_close:
                    torque_action[:, self.franka_hand_dof_index:self.franka_hand_dof_index + 2] = sim_config["robot"][
                        "fgripper_torque"]
                else:
                    torque_action[:, self.franka_hand_dof_index:self.franka_hand_dof_index + 2] = -sim_config["robot"][
                        "fgripper_torque"]

            self.isaac_gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(pos_action)
            )
            self.isaac_gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torque_action)
            )

            # Update viewer
            if not self.headless:
                self.isaac_gym.draw_viewer(self.viewer, self.sim, False)
                self.isaac_gym.sync_frame_time(self.sim)

            if self.record:
                self._record_video("front", env_idx)
                self._record_video("back", env_idx)
                self._record_video("left", env_idx)
                self._record_video("right", env_idx)

        self.env_steps += 1
        obs = []
        for i in range(self.num_envs):
            if self.record_obs[i]:
                self.observation[i].append(self.get_obs(i))
                obs.append(self.observation[i])

        if not obs:
            obs = None
        return obs,None, None, None

    def _record_video(self, camera, env_idx):
        img = torch.tensor(
            self.isaac_gym.get_camera_image(
                self.sim, self.envs[env_idx], self.camera_handles[camera][env_idx], gymapi.IMAGE_COLOR
            )
        ).to(self.device)
        img = img.reshape(self.img_size[1], self.img_size[0], 4)
        if not self.np_step_out:
            img = img.cpu().numpy().copy()
        if self.channel_first:
            img = img.transpose(0, 2, 3, 1)
        # img = img.squeeze()
        if camera == "front":
            self.video_writer1.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
        elif camera == "back":
            self.video_writer2.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
        elif camera == "left":
            self.video_writer3.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
        elif camera == "right":
            self.video_writer4.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))

    def _save_camera_input(self):
        """Saves camera images to png files for debugging."""
        root = "sim_camera"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(root).mkdir(exist_ok=True)

        for cam, handles in self.camera_handles.items():
            self.isaac_gym.write_camera_image_to_file(
                self.sim,
                self.envs[0],
                handles[0],
                gymapi.IMAGE_COLOR,
                f"{root}/{timestamp}_{cam}_sim.png",
            )

            self.isaac_gym.write_camera_image_to_file(
                self.sim,
                self.envs[0],
                handles[0],
                gymapi.IMAGE_DEPTH,
                f"{root}/{timestamp}_{cam}_sim_depth.png",
            )

    def _read_robot_state(self):
        joint_positions = self.dof_pos[:, self.franka_dof_index:self.franka_dof_index + 7]
        joint_velocities = self.dof_vel[:, self.franka_dof_index:self.franka_dof_index + 7]
        joint_torques = self.forces[:, self.franka_dof_index:self.franka_dof_index + 7]
        ee_pos, ee_quat = self.get_ee_pose_robot()
        for q in ee_quat:
            if q[3] < 0:
                q *= -1

        ee_pos_vel = self.rb_states[self.ee_idxs, 7:10]
        ee_ori_vel = self.rb_states[self.ee_idxs, 10:]

        gripper_width = self.gripper_width()

        robot_state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_pos_vel": ee_pos_vel,
            "ee_ori_vel": ee_ori_vel,
            "gripper_width": gripper_width,
        }
        return robot_state_dict

    def refresh(self):
        self.isaac_gym.simulate(self.sim)
        self.isaac_gym.fetch_results(self.sim, True)
        self.isaac_gym.step_graphics(self.sim)

        # Refresh tensors.
        self.isaac_gym.refresh_dof_state_tensor(self.sim)
        self.isaac_gym.refresh_dof_force_tensor(self.sim)
        self.isaac_gym.refresh_rigid_body_state_tensor(self.sim)
        self.isaac_gym.refresh_jacobian_tensors(self.sim)
        self.isaac_gym.refresh_mass_matrix_tensors(self.sim)
        self.isaac_gym.render_all_camera_sensors(self.sim)
        self.isaac_gym.start_access_image_tensors(self.sim)

    def init_ctrl(self):
        # Positional and velocity gains for robot control.
        kp = torch.tensor(sim_config["robot"]["kp"], device=self.device)
        kv = (
            torch.tensor(sim_config["robot"]["kv"], device=self.device)
            if sim_config["robot"]["kv"] is not None
            else torch.sqrt(kp) * 2.0
        )

        ee_pos, ee_quat = self.get_ee_pose_robot()

        for env_idx in range(self.num_envs):
            self.osc_ctrls.append(
                osc_factory(
                    real_robot=False,
                    ee_pos_current=ee_pos[env_idx],
                    ee_quat_current=ee_quat[env_idx],
                    init_joints=torch.tensor(
                        config["robot"]["reset_joints"], device=self.device
                    ),
                    kp=kp,
                    kv=kv,
                    mass_matrix_offset_val=[0.0, 0.0, 0.0],
                    position_limits=torch.tensor(
                        config["robot"]["position_limits"], device=self.device
                    ),
                    joint_kp=10,
                )
            )
        self.ctrl_started = True

    def get_ee_pose_robot(self):
        """Gets end-effector pose in world coordinate."""
        hand_pos = self.rb_states[self.ee_idxs, :3]
        hand_quat = self.rb_states[self.ee_idxs, 3:7]
        base_pos = self.rb_states[self.base_idxs, :3]
        base_quat = self.rb_states[self.base_idxs, 3:7]  # Align with world coordinate.

        return hand_pos - base_pos, hand_quat  # , hand_pos_2 - base_pos_2, hand_quat_2

    def get_ee_pose_world(self):
        """Gets end-effector pose in world coordinate."""
        hand_pos = self.rb_states[self.ee_idxs, :3]
        hand_quat = self.rb_states[self.ee_idxs, 3:7]

        return hand_pos, hand_quat

    def gripper_width(self):
        # zhp: need to change
        return self.dof_pos[:, self.franka_dof_index + 7:self.franka_dof_index + 8] + self.dof_pos[:,
                                                                                      self.franka_dof_index + 8:self.franka_dof_index + 9]

    def is_success(self):
        return [{"task": self.furnitures[env_idx].all_assembled()} for env_idx in range(self.num_envs)]

    def reset(self):
        # can also reset the full set of robots/parts, without applying torques and refreshing
        # self._reset_franka_all()
        # self._reset_parts_all()
        for i in range(self.num_envs):
            # if using ._reset_*_all(), can set reset_franka=False and reset_parts=False in .reset_env
            self.reset_env(i)

            # apply zero torque across the board and refresh in between each env reset (not needed if using ._reset_*_all())
            torque_action = torch.zeros_like(self.dof_pos)
            self.isaac_gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torque_action)
            )
            self.refresh()
        self.furniture.reset()

        self._reset_check()

        self.refresh()

        if self.save_camera_input:
            self._save_camera_input()

        return

    def reset_to(self, state):
        """Reset to a specific state.

        Args:
            state: List of observation dictionary for each environment.
        """
        for i in range(self.num_envs):
            self.reset_env_to(i, state[i])

    def reset_env(self, env_idx, reset_franka=True, reset_parts=True):
        """Resets the environment. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx: Environment index.
            reset_franka: If True, then reset the franka for this env
            reset_parts: If True, then reset the part poses for this env
        """
        self.furnitures[env_idx].reset()

        if reset_franka:
            self._reset_franka(env_idx)
        if reset_parts:
            self._reset_parts(env_idx)

        self._reset_hand(env_idx)
        # self._reset_base_table(env_idx)
        self.reset_obs(env_idx)
        self.env_steps[env_idx] = 0
        self.move_neutral = False

    def reset_env_to(self, env_idx, state):
        """Reset to a specific state. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx: Environment index.
            state: A dict containing the state of the environment.
        """
        self.furnitures[env_idx].reset()
        dof_pos = np.concatenate(
            [
                state["robot_state"]["joint_positions"],
                np.array([state["robot_state"]["gripper_width"] / 2] * 2),
            ],
        )
        self._reset_franka(env_idx, dof_pos)
        self._reset_parts(env_idx, state["parts_poses"])
        self.env_steps[env_idx] = 0
        self.move_neutral = False

    def _update_franka_dof_state_buffer(self, dof_pos=None):
        """
        Sets internal tensor state buffer for Franka actor
        """

        dof_pos = self.default_dof_pos if dof_pos is None else dof_pos

        # Views for self.dof_states (used with set_dof_state_tensor* function)
        self.dof_pos[:, self.franka_dof_index: self.franka_dof_index + self.franka_num_dofs] = torch.tensor(
            dof_pos, device=self.device, dtype=torch.float32
        )
        self.dof_vel[:, self.franka_dof_index: self.franka_dof_index + self.franka_num_dofs] = torch.tensor(
            [0] * len(self.default_dof_pos), device=self.device, dtype=torch.float32
        )

    def _reset_franka(self, env_idx, dof_pos=None):
        """
        Resets Franka actor within a single env. If calling multiple times,
        need to refresh in between calls to properly register individual env changes,
        and set zero torques on frankas across all envs to prevent the reset arms
        from moving while others are still being reset
        """
        self._update_franka_dof_state_buffer(dof_pos=dof_pos)

        # Update a single actor
        actor_idx = self.franka_actor_idxs_all_t[env_idx].reshape(1, 1)
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(actor_idx),
            len(actor_idx),
        )

    def _reset_franka_all(self, dof_pos=None):
        """
        Resets all Franka actors across all envs
        """
        self._update_franka_dof_state_buffer(dof_pos=dof_pos)

        # Update all actors across envs at once
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(self.franka_actor_idxs_all_t),
            len(self.franka_actor_idxs_all_t),
        )

    def _reset_parts(self, env_idx, parts_poses=None, skip_set_state=False):
        """Resets furniture parts to the initial pose.

        Args:
            env_idx (int): The index of the environment.
            parts_poses (np.ndarray): The poses of the parts. If None, the parts will be reset to the initial pose.
        """
        for part_idx, part in enumerate(self.furnitures[env_idx].parts):
            if part.name not in self.task_config["part_names"]:
                continue
            # Use the given pose.
            if parts_poses is not None:
                part_pose = parts_poses[part_idx * 7: (part_idx + 1) * 7]

                pos = part_pose[:3]
                ori = T.to_homogeneous(
                    [0, 0, 0], T.quat2mat(part_pose[3:])
                )  # Dummy zero position.
            else:
                pos, ori = self._get_reset_pose(part)

            part_pose_mat = self.table_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = gymapi.Transform()
            part_pose.p = gymapi.Vec3(
                part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
            )
            reset_ori = self.table_coord_to_sim_coord(ori)
            part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
            idxs = self.parts_handles[part.name]
            idxs = torch.tensor(idxs, device=self.device, dtype=torch.int32)
            self.root_pos[env_idx, idxs] = torch.tensor(
                [part_pose.p.x, part_pose.p.y, part_pose.p.z], device=self.device
            )
            self.root_quat[env_idx, idxs] = torch.tensor(
                [part_pose.r.x, part_pose.r.y, part_pose.r.z, part_pose.r.w],
                device=self.device,
            )
        if skip_set_state:
            # Set the value for the root state tensor, but don't call isaac gym function yet (useful when resetting all at once)
            # If skip_set_state == True, then must self.refresh() to register the isaac set_actor_root_state* function
            return

        # zhp: reset the base table
        base_table_idxs = torch.tensor(self.table_actor_index, device=self.device, dtype=torch.int32)
        self.root_pos[env_idx, base_table_idxs] = torch.tensor(
            [self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z], device=self.device
        )
        self.root_quat[env_idx, base_table_idxs] = torch.tensor(
            [self.table_pose.r.x, self.table_pose.r.y, self.table_pose.r.z, self.table_pose.r.w],
            device=self.device,
        )

        # Reset root state for actors in a single env
        part_actor_idxs = torch.tensor(self.part_actor_idx_by_env[env_idx] + [self.table_actor_index],
                                       device=self.device, dtype=torch.int32)
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(part_actor_idxs),
            len(part_actor_idxs),
        )

    def _reset_parts_all(self, parts_poses=None):
        """Resets ALL furniture parts to the initial pose.

        Args:
            parts_poses (np.ndarray): The poses of the parts. If None, the parts will be reset to the initial pose.
        """
        for env_idx in range(self.num_envs):
            self._reset_parts(env_idx, parts_poses=parts_poses, skip_set_state=True)

        # Reset root state for actors across all envs
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(self.part_actor_idxs_all_t),
            len(self.part_actor_idxs_all_t),
        )

    def _import_obstacle_front_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = "furniture/urdf/obstacle_front.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, obstacle_asset_file, asset_options
        )

    def _import_obstacle_side_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = "furniture/urdf/obstacle_side.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, obstacle_asset_file, asset_options
        )

    def _import_background_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        background_asset_file = "furniture/urdf/background.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, background_asset_file, asset_options
        )

    def _import_table_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 1
        asset_options.thickness = 0.001
        asset_options.fix_base_link = False
        table_asset_file = "furniture/urdf/table.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, table_asset_file, asset_options
        )

    def _import_fixed_table_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset_file = "furniture/urdf/table.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, table_asset_file, asset_options
        )

    def _import_franka_asset(self):
        self.franka_asset_file = (
            "franka_description_ros/franka_description/robots/franka_panda.urdf"
        )
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, self.franka_asset_file, asset_options
        )

    def _import_franka_hand_asset(self):
        self.franka_asset_file = (
            "franka_description_ros/franka_description/robots/franka_panda_hand_movable.urdf"
        )
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, self.franka_asset_file, asset_options
        )

    def _import_box_asset(self):
        if self.box_config is not None:
            size = self.box_config["size"]
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            return self.isaac_gym.create_box(self.sim, size[0], size[1], size[2], asset_options)
        else:
            return None

    def __del__(self):
        if not self.headless:
            self.isaac_gym.destroy_viewer(self.viewer)
        self.isaac_gym.destroy_sim(self.sim)

        if self.record:
            self.video_writer1.release()
            self.video_writer2.release()
            self.video_writer3.release()
            self.video_writer4.release()

    # zhp: functions
    def close_hand(self, cr=True):
        self.hand_close = cr

    def set_hand_transform(self, hand_pos, hand_ori):
        env = self.envs[0]

        self.franka_hand_pose = gymapi.Transform()
        self.franka_hand_pose.p = gymapi.Vec3(
            hand_pos[0], hand_pos[1], hand_pos[2]
        )
        self.franka_hand_pose.r = gymapi.Quat(*T.mat2quat(hand_ori[:3, :3]))

        # Set hand state
        env_idx = 0
        idxs = torch.tensor(self.franka_hand_actor_index, device=self.device, dtype=torch.int32)
        self.root_pos[env_idx, idxs] = torch.tensor(
            [self.franka_hand_pose.p.x, self.franka_hand_pose.p.y, self.franka_hand_pose.p.z], device=self.device
        )
        self.root_quat[env_idx, idxs] = torch.tensor(
            [self.franka_hand_pose.r.x, self.franka_hand_pose.r.y, self.franka_hand_pose.r.z,
             self.franka_hand_pose.r.w],
            device=self.device,
        )

        # Reset root state for actors in a single env
        hand_actor_idxs = torch.tensor([self.franka_hand_actor_index], device=self.device, dtype=torch.int32)
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(hand_actor_idxs),
            len(hand_actor_idxs),
        )

    def _reset_hand(self, env_idx):
        hand_ori = get_mat([0, 0, 0], [0, np.pi / 2, np.pi / 2])
        hand_pos = [0, 0.0, 1.5]
        self.franka_hand_pose = gymapi.Transform()
        self.franka_hand_pose.p = gymapi.Vec3(
            hand_pos[0], hand_pos[1], hand_pos[2]
        )

        idxs = torch.tensor(self.franka_hand_actor_index, device=self.device, dtype=torch.int32)
        self.root_pos[env_idx, idxs] = torch.tensor(
            [self.franka_hand_pose.p.x, self.franka_hand_pose.p.y, self.franka_hand_pose.p.z], device=self.device
        )
        self.root_quat[env_idx, idxs] = torch.tensor(
            [self.franka_hand_pose.r.x, self.franka_hand_pose.r.y, self.franka_hand_pose.r.z,
             self.franka_hand_pose.r.w],
            device=self.device,
        )

        # Reset root state for actors in a single env
        hand_actor_idxs = torch.tensor([self.franka_hand_actor_index], device=self.device, dtype=torch.int32)
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(hand_actor_idxs),
            len(hand_actor_idxs),
        )

    def _reset_base_table(self, env_idx):
        idxs = torch.tensor(self.table_actor_index, device=self.device, dtype=torch.int32)
        self.root_pos[env_idx, idxs] = torch.tensor(
            [self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z], device=self.device
        )
        self.root_quat[env_idx, idxs] = torch.tensor(
            [self.table_pose.r.x, self.table_pose.r.y, self.table_pose.r.z, self.table_pose.r.w],
            device=self.device,
        )

        # Reset root state for actors in a single env
        table_actor_idxs = torch.tensor([self.table_actor_index], device=self.device, dtype=torch.int32)
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(table_actor_idxs),
            len(table_actor_idxs),
        )
