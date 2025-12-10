import torch
import numpy as np
import furniture_bench.controllers.control_utils as C
from tqdm import trange

from tasks.utils import *
from tasks.task_config import all_task_config

task_name = "drawer_bottom"
task_config = all_task_config[task_name]
furniture_name = task_config["furniture_name"]
part1_name = task_config["part_names"][0]
part2_name = task_config["part_names"][1]
bias = 0.0
# part2_name = None


def reset_global(task_n):
    global task_name, task_config, furniture_name, part1_name, part2_name, bias
    task_name = task_n
    task_config = all_task_config[task_name]
    furniture_name = task_config["furniture_name"]
    part1_name = task_config["part_names"][0]
    part2_name = task_config["part_names"][1]

    if "1" in task_name:
        bias = 0.029

    elif "2" in task_name:
        bias = -0.029


# Path Planning
def pre_push(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 1.5
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part1_pose = env_states[0]
    part1_ori = part1_pose[:3, :3]
    part1_pos = part1_pose[:3, 3]
    part1_angles = rot_mat_to_angles_tensor(part1_ori, env.device)
    part1_forward = part1_angles[2].item()

    target_ori_1 = rot_mat_tensor(0, np.pi / 2 + np.pi / 3, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part1_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part1_pos
    target_pos_1[1] += 0.2 * np.cos(part1_forward) * factor
    target_pos_1[0] -= 0.2 * np.sin(part1_forward) * factor
    target_pos_1[2] += 0.000 * factor

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_push_closer(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 1.5
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part1_pose = env_states[0]
    part1_ori = part1_pose[:3, :3]
    part1_pos = part1_pose[:3, 3]
    part1_angles = rot_mat_to_angles_tensor(part1_ori, env.device)
    part1_forward = part1_angles[2].item()

    target_ori_1 = rot_mat_tensor(0, np.pi / 2 + np.pi / 3, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part1_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part1_pos
    target_pos_1[1] += 0.16 * np.cos(part1_forward) * factor
    target_pos_1[0] -= 0.16 * np.sin(part1_forward) * factor
    target_pos_1[2] += 0.000 * factor

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def push(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 1.5
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part1_pose = env_states[0]
    part1_ori = part1_pose[:3, :3]
    part1_pos = part1_pose[:3, 3]
    part1_angles = rot_mat_to_angles_tensor(part1_ori, env.device)
    part1_forward = part1_angles[2].item()

    target_ori_1 = rot_mat_tensor(0, np.pi / 2 + np.pi / 3, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part1_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part1_pos
    target_pos_1[1] += 0.09 * np.cos(part1_forward) * factor
    target_pos_1[0] -= 0.09 * np.sin(part1_forward) * factor
    target_pos_1[2] += 0.002 * factor

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]

    return target_ee_states, thresholds, False


def pre_grasp(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 3.0
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item()
    target_ori_1 = rot_mat_tensor(0, np.pi / 2, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part2_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part2_pos
    target_pos_1[1] += 0.2 * np.cos(part2_forward) * factor
    target_pos_1[0] -= 0.2 * np.sin(part2_forward) * factor

    target_pos_1[1] += bias * 3.0 * np.sin(part2_forward) * factor
    target_pos_1[0] += bias * 3.0 * np.cos(part2_forward) * factor

    target_pos_1[2] += 0.000 * factor
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_grasp_close(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 3.0
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item()
    target_ori_1 = rot_mat_tensor(0, np.pi / 2, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part2_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part2_pos
    target_pos_1[1] += 0.14 * np.cos(part2_forward) * factor
    target_pos_1[0] -= 0.14 * np.sin(part2_forward) * factor
    target_pos_1[1] += bias * 3.0 * np.sin(part2_forward) * factor
    target_pos_1[0] += bias * 3.0 * np.cos(part2_forward) * factor
    target_pos_1[2] -= 0.015 * factor
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pull(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor / 3.0
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item()
    target_ori_1 = rot_mat_tensor(0, np.pi / 2, 0, env.device)
    target_ori_1 = (
        rot_mat_tensor(0, 0, part2_forward - np.pi / 2, env.device) @ target_ori_1
    )
    target_pos_1 = part2_pos
    target_pos_1[1] += 0.35 * np.cos(part2_forward) * factor
    target_pos_1[0] -= 0.35 * np.sin(part2_forward) * factor

    target_pos_1[1] += bias * 3.0 * np.sin(part2_forward) * factor
    target_pos_1[0] += bias * 3.0 * np.cos(part2_forward) * factor

    target_pos_1[2] -= 0.015 * factor

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


func_map = {
    "close_gripper": close_gripper,
    "release_gripper": release_gripper,
    "pre_push": pre_push,
    "pre_push_closer": pre_push_closer,
    "push": push,
    "pre_grasp": pre_grasp,
    "pre_grasp_close": pre_grasp_close,
    "pull": pull,
}


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


def prepare(env, prev_target_ee_states):
    target_ee_states, result = act_phase(
        env, "pre_push", func_map, prev_target_ee_states
    )
    return target_ee_states, result


def prepare_r(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    target_ee_states, result = act_phase(
        env, "release_gripper", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "pre_grasp", func_map, prev_target_ee_states
    )
    return target_ee_states, result


def perform(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    target_ee_states, result = act_phase(
        env, "pre_push_closer", func_map, target_ee_states, slow=True
    )
    target_ee_states, result = act_phase(env, "push", func_map, target_ee_states)
    return True


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
    # target_ee_states, result = act_phase(env, "release_gripper", func_map, target_ee_states)

    return True


def perform_disassemble(env, prev_target_ee_states):
    # not implemented yet
    target_ee_states = prev_target_ee_states
    # zhp: Efficiency? seems like release_gripper takes a while
    target_ee_states, result = act_phase(
        env, "release_gripper", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(env, "pre_grasp", func_map, target_ee_states)
    if not result:
        print("Gripper Collision at Pre Grasp")
        return False
    target_ee_states, result = act_phase(
        env, "close_gripper", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(env, "pull", func_map, target_ee_states)
    return True
