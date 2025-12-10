import torch
import numpy as np
import furniture_bench.controllers.control_utils as C

from tasks.utils import *
from tasks.task_config import all_task_config

task_name = "desk_1_pull"
task_config = all_task_config[task_name]
furniture_name = task_config["furniture_name"]
part1_name = task_config["part_names"][0]
part2_name = task_config["part_names"][1]


y_angle = 0  # np.pi/6


def reset_global(task_n):
    global task_name, task_config, furniture_name, part1_name, part2_name
    task_name = task_n
    task_config = all_task_config[task_name]
    furniture_name = task_config["furniture_name"]
    part1_name = task_config["part_names"][0]
    part2_name = task_config["part_names"][1]


def pre_grasp(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 1.5
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item() + np.pi
    target_ori_1 = rot_mat_tensor(0, np.pi / 2 + y_angle, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part2_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part2_pos
    target_pos_1[1] += 0.155 * np.cos(part2_forward) * factor
    target_pos_1[0] -= 0.155 * np.sin(part2_forward) * factor
    target_pos_1[2] += 0.0 * factor
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_grasp_close(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 1.5
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item() + np.pi
    target_ori_1 = rot_mat_tensor(0, np.pi / 2 + y_angle, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part2_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part2_pos
    target_pos_1[1] += 0.095 * np.cos(part2_forward) * factor
    target_pos_1[0] -= 0.095 * np.sin(part2_forward) * factor
    target_pos_1[2] += 0.0 * factor
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pull(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 1.5
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item() + np.pi
    target_ori_1 = rot_mat_tensor(0, np.pi / 2 + y_angle, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part2_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part2_pos
    target_pos_1[1] += 0.2 * np.cos(part2_forward) * factor
    target_pos_1[0] -= 0.2 * np.sin(part2_forward) * factor
    target_pos_1[2] += 0.015 * factor

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def release_gripper(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (start_pos_1, start_quat_1, gripper)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, True


def close_gripper(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    gripper_action = 1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (start_pos_1, start_quat_1, gripper)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, True


def act_phase(env, phase, func_map, last_target_ee_states=None, slow=False):
    rb_states = env.rb_states
    part_idxs = env.part_idxs

    part2_pose = C.to_homogeneous(
        rb_states[part_idxs[part2_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
    )

    part1_pose = C.to_homogeneous(
        rb_states[part_idxs[part1_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
    )

    if last_target_ee_states is not None:
        ee_pos_1, ee_quat_1, gripper_1 = last_target_ee_states[0]
    else:
        ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
        gripper_1 = env.last_grasp_1

    ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
    gripper_1 = gripper_1.squeeze()

    start_ee_states = [(ee_pos_1, ee_quat_1, gripper_1)]
    env_states = [part1_pose, part2_pose]

    target_ee_states, thresholds, is_gripper = func_map[phase](
        env, start_ee_states, env_states
    )
    if phase == "pre_push":
        pose_spend_time = 60
    elif phase == "pre_push_closer":
        pose_spend_time = 10
    else:
        pose_spend_time = 40
    result = reach_target(
        env,
        target_ee_states,
        thresholds,
        is_gripper,
        slow=slow,
        pose_spend_time=pose_spend_time,
    )

    return target_ee_states, result


func_map = {
    "close_gripper": close_gripper,
    "release_gripper": release_gripper,
    "pre_grasp": pre_grasp,
    "pre_grasp_close": pre_grasp_close,
    "pull": pull,
}


def prepare(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    target_ee_states, result = act_phase(
        env, "release_gripper", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "pre_grasp", func_map, prev_target_ee_states
    )
    return target_ee_states, result


def perform_r(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states

    target_ee_states, result = act_phase(
        env, "pre_grasp_close", func_map, target_ee_states
    )
    if not result:
        print("Gripper Collision at Pre Grasp")
        return False
    target_ee_states, result = act_phase(
        env, "close_gripper", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(env, "pull", func_map, target_ee_states)
    target_ee_states, result = act_phase(
        env, "pull", func_map, target_ee_states, slow=True
    )

    return True
