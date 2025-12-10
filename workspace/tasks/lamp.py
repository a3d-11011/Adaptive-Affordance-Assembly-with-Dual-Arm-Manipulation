import torch
import numpy as np
import furniture_bench.controllers.control_utils as C

from utils.rotation import euler_from_deltaR
from tasks.utils import *
from tasks.task_config import all_task_config

task_name = "lamp"
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
    control_hand(env, close=False)
    part2_pose = env.rb_states[env.part_idxs[part2_name]][0]
    part2_pos = part2_pose[:3]
    part2_ori = part2_pose[3:7]
    part2_forward = rot_mat_to_angles_tensor(C.quat2mat(part2_ori), env.device)[
        2
    ].item()
    hand_pos = part2_pos.cpu().numpy()
    hand_pos[2] += 0.105
    hand_ori = rot_mat([np.pi, 0, part2_forward - np.pi / 2], hom=True)[:3, :3]
    env.set_hand_transform(hand_pos, hand_ori)
    control_hand(env, close=True)
    hand_pos[2] += 0.1
    smooth_hand_transform(env, hand_pos, hand_ori)
    hand_pos[1] += 0.0
    hand_ori = rot_mat(
        [np.pi, -np.pi / 2 + np.pi * 10 / 180, part2_forward - np.pi / 2], hom=True
    )[:3, :3]
    smooth_hand_transform(env, hand_pos, hand_ori)


def grasp_hood_0(env, start_ee_states, env_states):
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3, 3]
    target_pos_1[1] += 0.05
    target_pos_1[2] += 0.12
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def grasp_hood_1(env, start_ee_states, env_states):
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3, 3]
    target_pos_1[1] += 0.05
    target_pos_1[2] += 0.07
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper)
    ]  # (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def grasp_hood_2(env, start_ee_states, env_states):
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3, 3]
    target_pos_1[1] += 0.028
    target_pos_1[2] += 0.07
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = 1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper)
    ]  # (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def grasp_hood_3(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    target_pos_1 = start_pos_1[:3]
    target_pos_1[0] = 0
    target_pos_1[1] = 0
    target_pos_1[2] += 0.2

    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]

    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = 1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper)
    ]  # (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def move_to_(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    part2_pose = env_states[1]
    pos, ori = find_part2_pose(env)
    pos_move = torch.tensor(pos, device=part2_pose.device) - part2_pose[:3, 3]
    target_pos_1 = start_pos_1 + pos_move

    target_pos_1[0] += 0.0075
    target_pos_1[1] -= 0.0035

    target_pos_1[2] += 0.06
    if part2_name == "lamp_hood":
        target_pos_1[2] += 0.14

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
    target_pos_1[2] += 0.013

    # start_ori= C.quat2mat(start_quat_1)
    # target_ori_1= rot_mat_tensor(np.pi/2, 0, 0, env.device)[:3, :3]
    # dR= torch.matmul(part2_pose[:3,:3],target_ori_1.T)
    # # dR= torch.matmul(part2_pose[:3,:3],torch.tensor(ori[:3,:3],device=part2_pose.device,dtype=torch.float32).T)
    # da,db,dc= euler_from_deltaR(dR)
    # dR_xy=rot_mat_tensor(da.cpu(),db.cpu(),0, env.device)[:3,:3]

    # if part2_name != "lamp_hood":
    #     # target_ori_1= torch.matmul(dR, target_ori_1)
    #     target_ori_1 = torch.matmul( dR_xy,start_ori.to(env.device))
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)

    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


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


# Path Planning
def pre_grasp_z(env, start_ee_states, env_states):
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3, 3]
    target_pos_1[2] += 0.2
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pre_grasp_xy(env, start_ee_states, env_states):
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3, 3]
    target_pos_1[2] += 0.15
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
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3, 3]
    target_pos_1[2] += 0.07
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [
        (target_pos_1, target_quat_1, gripper_1)
    ]  # , (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def screw(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    # target_ori_1 = rot_mat_tensor(np.pi, 0, -np.pi / 2 - np.pi / 36, env.device)[
    #     :3, :3
    # ]
    start_rot_mat = C.quat2mat(start_quat_1)
    start_forward = rot_mat_to_angles_tensor(start_rot_mat, env.device)[2].item()
    target_ori_1 = rot_mat_tensor(
        np.pi, 0, start_forward - np.pi / 2 - np.pi / 72, env.device
    )[:3, :3]
    target_pos_1 = (start_pos_1)[:3]
    target_pos_1[2] += 0.0001
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, 0.3), (None, None)]
    return target_ee_states, thresholds, False


def rev_screw(env, start_ee_states, env_states):
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    target_ori_1 = rot_mat_tensor(np.pi, 0, np.pi / 2 + np.pi / 36, env.device)[:3, :3]
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
    "grasp_hood_0": grasp_hood_0,
    "grasp_hood_1": grasp_hood_1,
    "grasp_hood_2": grasp_hood_2,
    "grasp_hood_3": grasp_hood_3,
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


def pick_up_hood(env, prev_target_ee_states):
    target_ee_states, result = act_phase(
        env, "grasp_hood_0", func_map, prev_target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "grasp_hood_1", func_map, prev_target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "grasp_hood_2", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "close_gripper", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "grasp_hood_3", func_map, target_ee_states
    )
    target_ee_states, result = act_phase(env, "move_to_", func_map, target_ee_states)
    target_ee_states, result = act_phase(env, "insert_", func_map, target_ee_states)
    target_ee_states, result = act_phase(
        env, "release_gripper", func_map, target_ee_states
    )


def prepare(env, prev_target_ee_states):
    target_ee_states, result = act_phase(
        env, "pre_grasp_z", func_map, prev_target_ee_states
    )
    target_ee_states, result = act_phase(
        env, "pre_grasp_xy", func_map, prev_target_ee_states
    )
    return target_ee_states, result


def perform(env, prev_target_ee_states):

    target_ee_states = prev_target_ee_states
    for i in range(3):
        # zhp: Efficiency? seems like release_gripper takes a while
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
        target_ee_states, result = act_phase(env, "screw", func_map, target_ee_states)

    return True


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
    for i in range(5):
        # zhp: Efficiency? seems like release_gripper takes a while
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

    return True
