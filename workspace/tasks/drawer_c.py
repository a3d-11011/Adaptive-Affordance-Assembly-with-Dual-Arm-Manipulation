import torch
import numpy as np
import furniture_bench.controllers.control_utils as C

from tasks.utils import *
from tasks.task_config import all_task_config

task_name = "drawer_c"
task_config = all_task_config[task_name]
furniture_name = task_config["furniture_name"]
part1_name = task_config["part_names"][0]
last_length = 0.0


def reset_global(task_n):
    global task_name, task_config, furniture_name, part1_name, part2_name
    task_name = task_n
    task_config = all_task_config[task_name]
    furniture_name = task_config["furniture_name"]
    part1_name = task_config["part_names"][0]


# Path Planning


def get_now_target(env):
    factor = env.furniture_scale_factor
    rb_states = env.rb_states
    part_idxs = env.part_idxs

    part1_pose = C.to_homogeneous(
        rb_states[part_idxs[part1_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
    )
    part1_ori = part1_pose[:3, :3]
    part1_pos = part1_pose[:3, 3]
    part1_angles = rot_mat_to_angles_tensor(part1_ori, env.device)
    part1_forward = part1_angles[2].item()
    target_pos_1 = part1_pos
    target_pos_1[1] -= 0.049 * np.sin(part1_forward) * factor
    target_pos_1[0] += 0.049 * np.cos(part1_forward) * factor
    target_pos_1[1] += last_length * np.sin(part1_forward) * factor
    target_pos_1[0] += last_length * np.cos(part1_forward) * factor
    target_pos_1[2] += 0.134 * factor
    return target_pos_1


def pre_grasp(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part1_pose = env_states[0]
    part1_ori = part1_pose[:3, :3]
    part1_pos = part1_pose[:3, 3]
    part1_angles = rot_mat_to_angles_tensor(part1_ori, env.device)
    part1_forward = part1_angles[2].item()
    part1_forward += np.pi / 2+np.pi
    # print("Angles: ", part1_angles.cpu().numpy()/np.pi*180)
    # print("Forward: ", part1_forward/np.pi*180)

    target_ori_1 = rot_mat_tensor(np.pi, 0.0, part1_forward , env.device)

    target_pos_1 = part1_pos
    target_pos_1[1] += 0.049 * np.cos(part1_forward) * factor
    target_pos_1[0] -= 0.049 * np.sin(part1_forward) * factor

    target_pos_1[1] += last_length * np.sin(part1_forward) * factor
    target_pos_1[0] += last_length * np.cos(part1_forward) * factor
    
    target_pos_1[2] += 0.03 * factor
    
    # target_pos_1 = torch.tensor([0.4, 0.4, 0.6], device=env.device)
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_grasp_closer(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part1_pose = env_states[0]
    part1_ori = part1_pose[:3, :3]
    part1_pos = part1_pose[:3, 3]
    part1_angles = rot_mat_to_angles_tensor(part1_ori, env.device)
    part1_forward = part1_angles[2].item()
    part1_forward += np.pi / 2+np.pi
    # print("Angles: ", part1_angles.cpu().numpy()/np.pi*180)
    # print("Forward: ", part1_forward/np.pi*180)

    target_ori_1 = rot_mat_tensor(np.pi, 0.0, part1_forward, env.device)

    target_pos_1 = part1_pos
    target_pos_1[1] += 0.049 * np.cos(part1_forward) * factor
    target_pos_1[0] -= 0.049 * np.sin(part1_forward) * factor

    target_pos_1[1] += last_length * np.sin(part1_forward) * factor
    target_pos_1[0] += last_length * np.cos(part1_forward) * factor
    
    target_pos_1[2] += 0.01 * factor
    # target_pos_1 = torch.tensor([0.4, 0.4, 0.6], device=env.device)
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def lift_up(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    target_pos_1 = start_pos_1
    target_pos_1[2] += 0.05
    target_ee_states = [(target_pos_1, start_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    hand_pose = env.franka_hand_pose
    p = hand_pose.p
    r = hand_pose.r
    target_hand_pos = np.array([p.x, p.y, p.z + 0.05])
    cur_q = np.array([r.x, r.y, r.z, r.w])
    set_hand_transform_with_rt(env, target_hand_pos, cur_q)
    return target_ee_states, thresholds, False


def lift_down(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    target_pos_1 = start_pos_1
    target_pos_1[2] -= 0.01
    target_ee_states = [(target_pos_1, start_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    hand_pose = env.franka_hand_pose
    p = hand_pose.p
    r = hand_pose.r
    target_hand_pos = np.array([p.x, p.y, p.z - 0.01])
    cur_q = np.array([r.x, r.y, r.z, r.w])
    set_hand_transform_with_rt(env, target_hand_pos, cur_q)
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
    "pre_grasp": pre_grasp,
    "pre_grasp_closer": pre_grasp_closer,
    "lift_up": lift_up,
    "lift_down": lift_down,
}


def act_phase(env, phase, func_map, last_target_ee_states=None):
    rb_states = env.rb_states
    part_idxs = env.part_idxs

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
    env_states = [part1_pose]

    target_ee_states, thresholds, is_gripper = func_map[phase](
        env, start_ee_states, env_states
    )
    # if phase == "lift_up":
    #     result = reach_target(env, target_ee_states, thresholds, is_gripper,pose_spend_time=20)
    # else:
    result = reach_target(
        env, target_ee_states, thresholds, is_gripper, slow=True, pose_spend_time=60
    )

    return target_ee_states, result


def reset_random_length():
    global last_length
    last_length = np.random.uniform(-0.02, 0.02)


def prepare(env, prev_target_ee_states):
    target_ee_states, result = act_phase(
        env, "pre_grasp", func_map, prev_target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "pre_grasp_closer", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "close_gripper", func_map, target_ee_states
    )

    return target_ee_states, result


def perform(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    target_ee_states, result = act_phase(env, "pre_grasp_closer", func_map, target_ee_states)
    target_ee_states, result = act_phase(
        env, "close_gripper", func_map, target_ee_states
    )
    for i in range(3):
        target_ee_states, result = act_phase(env, "lift_up", func_map, target_ee_states)
    return True


def perform_r(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    for i in range(3):
        target_ee_states, result = act_phase(
            env, "lift_down", func_map, target_ee_states
        )
    return True


def set_hand_transform(env):
    factor = env.furniture_scale_factor / 2.5
    rb_states = env.rb_states
    part_idxs = env.part_idxs
    part1_pose = C.to_homogeneous(
        rb_states[part_idxs[part1_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
    )
    part1_ori = part1_pose[:3, :3]
    part1_pos = part1_pose[:3, 3]
    part1_angles = rot_mat_to_angles_tensor(part1_ori, env.device)
    part1_forward = part1_angles[2].item() + np.pi
    target_ori_1 = rot_mat_tensor(np.pi / 2 - np.pi / 18, 0.0, 0, env.device)
    target_ori_1 = rot_mat_tensor(0, 0, part1_forward, env.device) @ target_ori_1
    target_pos_1 = part1_pos
    target_pos_1[1] += 0.3 * np.cos(part1_forward) * factor
    target_pos_1[0] -= 0.3 * np.sin(part1_forward) * factor
    target_pos_1[1] += last_length * np.sin(part1_forward) * factor
    target_pos_1[0] += last_length * np.cos(part1_forward) * factor
    target_pos_1[2] += 0.125 * factor
    env.set_hand_transform(target_pos_1.cpu().numpy(), target_ori_1.cpu().numpy())
