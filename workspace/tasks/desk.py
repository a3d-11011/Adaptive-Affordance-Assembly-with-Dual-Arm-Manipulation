import torch
import numpy as np
import furniture_bench.controllers.control_utils as C

from tasks.utils import *
from tasks.task_config import all_task_config

task_name = "desk"
task_config = all_task_config[task_name]
furniture_name = task_config["furniture_name"]
part1_name = task_config["part_names"][0]
part2_name = task_config["part_names"][1]


def reset_global(task_n):
    global task_name, task_config, furniture_name, part1_name, part2_name
    task_name = task_n
    task_config = all_task_config[task_name]
    furniture_name = task_config["furniture_name"]
    part1_name = task_config["part_names"][0]
    part2_name = task_config["part_names"][1]


def reset_part2_global(part2_n):
    global task_name, task_config, furniture_name, part1_name, part2_name
    task_config = all_task_config[task_name]
    furniture_name = task_config["furniture_name"]
    part1_name = task_config["part_names"][0]
    part2_name = part2_n


def hand_pick(env):
    part2_pose = env.rb_states[env.part_idxs[part2_name]][0]
    part2_pos = part2_pose[:3]
    part2_ori = part2_pose[3:7]
    part2_forward = rot_mat_to_angles_tensor(C.quat2mat(part2_ori), env.device)[
        2
    ].item()
    hand_pos = part2_pos.cpu().numpy()
    hand_pos[2] += 0.11
    hand_ori = rot_mat([np.pi, 0, part2_forward - np.pi / 2], hom=True)[:3, :3]
    env.set_hand_transform(hand_pos, hand_ori)
    control_hand(env, close=True)
    hand_pos[2] += 0.1
    smooth_hand_transform(env, hand_pos, hand_ori)
    hand_pos[1] += 0.1
    hand_ori = rot_mat([np.pi, np.pi / 2, part2_forward - np.pi / 2], hom=True)[:3, :3]
    smooth_hand_transform(env, hand_pos, hand_ori)


def pre_pick(env, start_ee_states, env_states):
    print("part2_name", part2_name)

    part2_pose = env_states[1]
    part2_pos = part2_pose[:3, 3]
    part2_forward = rot_mat_to_angles_tensor(part2_pose[:3, :3], env.device)[2].item()

    target_ori_1 = rot_mat_tensor(np.pi, 0, part2_forward - np.pi / 2, env.device)[
        :3, :3
    ]
    target_pos_1 = part2_pose[:3, 3]
    target_pos_1[1] -= 0.05 * np.cos(part2_forward)
    target_pos_1[0] += 0.05 * np.sin(part2_forward)
    target_pos_1[2] += 0.1
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_pick_close(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    part2_pose = env_states[1]
    part2_pos = part2_pose[:3, 3]
    part2_forward = rot_mat_to_angles_tensor(part2_pose[:3, :3], env.device)[2].item()
    target_ori_1 = rot_mat_tensor(np.pi, 0, part2_forward - np.pi / 2, env.device)[
        :3, :3
    ]
    target_pos_1 = part2_pose[:3, 3]
    target_pos_1[1] -= 0.05 * np.cos(part2_forward)
    target_pos_1[0] += 0.05 * np.sin(part2_forward)
    target_pos_1[2] += 0.0
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_pick_up(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    target_pos_1 = start_pos_1[:3]
    target_pos_1[2] += 0.2

    target_quat_1 = start_quat_1

    target_ee_states = [
        (target_pos_1, target_quat_1, gripper_1)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def rot_(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    euler = T.quat2euler(start_quat_1.cpu().numpy())
    euler[0] = 0
    euler[1] = np.pi / 2
    euler[2] = 0
    target_ori_1 = rot_mat_tensor(euler[0], euler[1], euler[2], env.device)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(start_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def move_to_(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    part2_pose = env_states[1]
    pos, ori = find_part2_pose(env)
    pos_move = torch.tensor(pos, device=part2_pose.device) - part2_pose[:3, 3]
    target_pos_1 = start_pos_1 + pos_move

    if part2_pose == "desk_leg3":
        target_pos_1[0] += 0.0075
        target_pos_1[1] -= 0.0035
    # elif
    # part2_name == "desk_leg4":
    else:
        target_pos_1[0] += 0.0075
        target_pos_1[1] -= 0.0035

    target_pos_1[2] += 0.05
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def insert_(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    part2_pose = env_states[1]
    pos, ori = find_part2_pose(env)
    pos_move = torch.tensor(pos, device=part2_pose.device) - part2_pose[:3, 3]
    target_pos_1 = start_pos_1 + pos_move
    target_pos_1[0] += 0.0075
    target_pos_1[1] -= 0.0035
    target_pos_1[2] += 0.0105
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


# Path Planning
def pre_grasp_z(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    target_pos_1 = start_pos_1[:3]
    target_pos_1[2] += 0.11
    target_quat_1 = start_quat_1
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper_1)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_grasp_xy(env, start_ee_states, env_states):

    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3, 3]
    target_pos_1[2] += 0.1
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_grasp(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    leg_pose = env_states[1]
    leg_forward = rot_mat_to_angles_tensor(leg_pose[:3, :3], env.device)[2] / np.pi
    leg_faces = [((leg_forward + i * 0.5 + 1) % 2) - 1 for i in range(4)]
    abs_diff = [abs(x) for x in leg_faces]
    min_index = abs_diff.index(min(abs_diff))
    selected_face = leg_faces[min_index].item()

    target_ori_1 = rot_mat_tensor(np.pi, 0, selected_face * np.pi, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3, 3]
    target_pos_1[2] += 0.05
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper_1)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def screw(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    start_rot_mat = C.quat2mat(start_quat_1)
    start_forward = rot_mat_to_angles_tensor(start_rot_mat, env.device)[2].item()
    target_ori_1 = rot_mat_tensor(
        np.pi, 0, start_forward - np.pi / 2 - np.pi / 72, env.device
    )[:3, :3]
    target_pos_1 = (start_pos_1)[:3]
    # target_pos_1[2] -= 0.005
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper_1)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, 0.3), (None, None)]
    return target_ee_states, thresholds, False


def rev_screw(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    start_rot_mat = C.quat2mat(start_quat_1)
    start_forward = rot_mat_to_angles_tensor(start_rot_mat, env.device)[2].item()
    target_ori_1 = rot_mat_tensor(
        np.pi, 0, start_forward + np.pi / 2 + np.pi / 36, env.device
    )[:3, :3]
    target_pos_1 = (start_pos_1)[:3]
    target_pos_1[2] -= 0.005
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, 0.3), (None, None)]
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


def find_part2_pose(env):

    from furniture_bench.assemble_config import config
    from furniture_bench.utils.pose import get_mat

    default_assembled_pose = config["furniture"][env.furniture_name][part2_name][
        "default_assembled_pose"
    ]
    scaled_default_assembled_pose = default_assembled_pose.copy()
    scaled_default_assembled_pose[:3, 3] *= env.furniture_scale_factor

    part1_now_pose = env.rb_states[env.part_idxs[part1_name]][0]
    part1_pos = part1_now_pose[:3].cpu().numpy()
    part1_ori = part1_now_pose[3:7]
    part1_ori = C.quat2mat(part1_ori).cpu().numpy()
    part1_pose = get_mat(part1_pos, part1_ori)

    part2_pose = part1_pose @ scaled_default_assembled_pose
    pos = part2_pose[:3, 3]
    ori = T.to_hom_ori(part2_pose[:3, :3])

    return pos, ori


func_map = {
    "close_gripper": close_gripper,
    "release_gripper": release_gripper,
    "pre_grasp": pre_grasp,
    "screw": screw,
    "pre_grasp_xy": pre_grasp_xy,
    "pre_grasp_z": pre_grasp_z,
    "rev_screw": rev_screw,
    "move_to_": move_to_,
    "insert_": insert_,
}


def act_phase(env, phase, func_map, last_target_ee_states=None):
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
    if phase == "insert_":
        result = reach_target(env, target_ee_states, thresholds, is_gripper, slow=True)
    else:
        result = reach_target(env, target_ee_states, thresholds, is_gripper)

    return target_ee_states, result


def prepare_pick(env, prev_target_ee_states):
    hand_pick(env)
    target_ee_states, result = act_phase(
        env, "pre_grasp_z", func_map, prev_target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "pre_grasp_xy", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(env, "pre_grasp", func_map, target_ee_states)

    target_ee_states, result = act_phase(
        env, "close_gripper", func_map, target_ee_states
    )
    env.close_hand(False)
    env.set_hand_transform(np.array([0, 0, 1.5]), rot_mat([0, 0, 0]))
    target_ee_states, result = act_phase(env, "move_to_", func_map, target_ee_states)
    target_ee_states, result = act_phase(env, "insert_", func_map, target_ee_states)
    target_ee_states, result = act_phase(env, "screw", func_map, target_ee_states)
    target_ee_states, result = act_phase(
        env, "release_gripper", func_map, target_ee_states
    )
    return target_ee_states, result


def prepare(env, prev_target_ee_states):
    target_ee_states, result = act_phase(
        env, "pre_grasp_z", func_map, prev_target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "pre_grasp_xy", func_map, prev_target_ee_states
    )
    target_ee_states, result = act_phase(
            env, "pre_grasp", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(
            env, "close_gripper", func_map, target_ee_states
    )

    return target_ee_states, result

def screw_(env, prev_target_ee_states):
    target_ee_states=prev_target_ee_states
    target_ee_states, result = act_phase(env, "screw", func_map, target_ee_states)
    return True

def perform(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    for i in range(4):
        target_ee_states, result = act_phase(
            env, "release_gripper", func_map, target_ee_states
        )

        # twice to adjust the position
        target_ee_states, result = act_phase(
            env, "pre_grasp_xy", func_map, target_ee_states
        )
        target_ee_states, result = act_phase(
            env, "pre_grasp_xy", func_map, target_ee_states
        )
        if not result:
            print("Gripper Collision at Pre Grasp XY")
            return False
        target_ee_states, result = act_phase(
            env, "pre_grasp", func_map, target_ee_states
        )
        if not result:
            print("Gripper Collision at Pre Grasp")
            return False
        target_ee_states, result = act_phase(
            env, "close_gripper", func_map, target_ee_states
        )
        target_ee_states, result = act_phase(env, "screw", func_map, target_ee_states)

    return True


def perform_(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    for i in range(3):

        # twice to adjust the position
        target_ee_states, result = act_phase(
            env, "pre_grasp_xy", func_map, target_ee_states
        )
        target_ee_states, result = act_phase(
            env, "pre_grasp_xy", func_map, target_ee_states
        )
        if not result:
            print("Gripper Collision at Pre Grasp XY")
            return False
        target_ee_states, result = act_phase(
            env, "pre_grasp", func_map, target_ee_states
        )
        if not result:
            print("Gripper Collision at Pre Grasp")
            return False
        target_ee_states, result = act_phase(
            env, "close_gripper", func_map, target_ee_states
        )
        target_ee_states, result = act_phase(env, "screw", func_map, target_ee_states)
        target_ee_states, result = act_phase(
            env, "release_gripper", func_map, target_ee_states
        )


def perform_r(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    for i in range(2):
        target_ee_states, result = act_phase(
            env, "release_gripper", func_map, target_ee_states
        )

        # twice to adjust the position
        target_ee_states, result = act_phase(
            env, "pre_grasp_xy", func_map, target_ee_states
        )
        target_ee_states, result = act_phase(
            env, "pre_grasp_xy", func_map, target_ee_states
        )
        if not result:
            print("Gripper Collision at Pre Grasp XY")
            return False
        target_ee_states, result = act_phase(
            env, "pre_grasp", func_map, target_ee_states
        )
        if not result:
            print("Gripper Collision at Pre Grasp")
            return False
        target_ee_states, result = act_phase(
            env, "close_gripper", func_map, target_ee_states
        )
        target_ee_states, result = act_phase(
            env, "rev_screw", func_map, target_ee_states
        )

    return True


def perform_disassemble(env, prev_target_ee_states):

    target_ee_states = prev_target_ee_states
    for i in range(4):
        target_ee_states, result = act_phase(
            env, "pre_grasp_xy", func_map, prev_target_ee_states
        )
        target_ee_states, result = act_phase(
            env, "release_gripper", func_map, target_ee_states
        )
        target_ee_states, result = act_phase(
            env, "pre_grasp", func_map, target_ee_states
        )
        if not result:
            print("Gripper Collision at Pre Grasp")
            return False
        target_ee_states, result = act_phase(
            env, "close_gripper", func_map, target_ee_states
        )
        target_ee_states, result = act_phase(
            env, "rev_screw", func_map, target_ee_states
        )

    target_ee_states, result = act_phase(
        env, "release_gripper", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "pre_grasp_xy", func_map, prev_target_ee_states
    )

    return True
