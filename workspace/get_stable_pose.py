import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
# import furniture_bench
import gym

import numpy as np
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import *
import isaacgym
from isaacgym import gymapi, gymtorch
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import trange
import importlib

from utils.rotation import rot_mat_to_angles
from tasks.utils import *
from tasks.task_config import all_task_config

DEBUG = False


def disassemble(task_config):
    furniture_name = task_config["furniture_name"]
    part1_name = task_config["part_names"][0]
    part2_name = task_config["part_names"][1]

    env = gym.make(
        "dual-franka-hand-v2",
        furniture=furniture_name,
        num_envs=1,
        # record=True,
        resize_img=False,
        assembled=True,
        set_friction=False,
        task_config=task_config,
    )

    env.reset()
    env.refresh()
    move_franka_away(env)
    # Start to perform the task
    rb_states = env.rb_states
    part_idxs = env.part_idxs
    module_name = task_config["task_name"].split("_")[0]
    # module_name = task_config["task_name"]
    task_module = importlib.import_module(f"tasks.{task_name}")

    task_module.task_name = task_config["task_name"]
    task_module.task_config = task_config
    task_module.furniture_name = furniture_name
    task_module.part1_name = part1_name
    task_module.part2_name = part2_name

    prepare = getattr(task_module, "prepare")
    perform_disassemble = getattr(task_module, "perform_disassemble")

    # prepare(env)
    target_ee_states = None
    #     target_ee_states, result = prepare(env, target_ee_states)

    part2_pose = C.to_homogeneous(
        rb_states[part_idxs[part2_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
    )

    all_finished = perform_disassemble(env, target_ee_states)

    wait(env)
    part1_pose = C.to_homogeneous(
        rb_states[part_idxs[part1_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
    )
    part2_pose = C.to_homogeneous(
        rb_states[part_idxs[part2_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
    )
    # Compute relative pose
    # Use torch to compute relative pose
    relative_pose = torch.linalg.inv(part1_pose) @ part2_pose
    relative_pose = relative_pose.cpu().numpy()
    #     relative_pose = get_mat([-0.00026228, -0.03447104, 0.04 ], [0, 0, 0])
    #     relative_pose[:3,3] = relative_pose[:3,3] *1.5
    print(relative_pose[:3, 3] / 1.5)
    print(relative_pose[:3, 3])
    print(np.degrees(rot_mat_to_angles(relative_pose[:3, :3])))
    # Save relative_pose
    relative_pose_path = os.path.join(
        BASE_DIR,
        "tasks",
        "relative_poses",
        f"{furniture_name}_{part1_name}_{part2_name}.npy",
    )
    if not DEBUG:
        np.save(relative_pose_path, relative_pose)

    env.refresh()


if __name__ == "__main__":
    task_name = "oval_table"
    task_config = all_task_config[task_name]
    disassemble(task_config)
