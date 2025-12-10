import torch
import numpy as np
import furniture_bench.controllers.control_utils as C
from tqdm import trange

from tasks.utils import *
from tasks.task_config import all_task_config

task_name = "cabinet_door_left"
task_config = all_task_config[task_name]
furniture_name = task_config["furniture_name"]
part1_name = task_config["part_names"][0]
part2_name = task_config["part_names"][1]
# part2_name = None


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


def pick_(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item()
    print("Angles: ", part2_angles / np.pi * 180)
    print("Forward: ", part2_forward / np.pi * 180)

    target_ori_1 = rot_mat_tensor(np.pi, 0, part2_forward, env.device)

    target_pos_1 = part2_pos
    target_pos_1[0] += 0.037 * np.sin(part2_forward) * factor
    target_pos_1[1] -= 0.037 * np.cos(part2_forward) * factor
    # target_pos_1[0] += 0.01 * np.sin(part2_forward) * factor
    # target_pos_1[1] -= 0.01 * np.cos(part2_forward) * factor
    target_pos_1[2] += 0.1 * factor

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pick_closer(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item()
    print("Angles: ", part2_angles / np.pi * 180)
    print("Forward: ", part2_forward / np.pi * 180)
    target_ori_1 = rot_mat_tensor(np.pi, 0, part2_forward, env.device)

    target_pos_1 = part2_pos
    target_pos_1[0] += 0.037 * np.sin(part2_forward) * factor
    target_pos_1[1] -= 0.037 * np.cos(part2_forward) * factor
    # target_pos_1[0] += 0.012 * np.sin(part2_forward) * factor
    # target_pos_1[1] -= 0.012 * np.cos(part2_forward) * factor
    target_pos_1[2] += 0.012 * factor

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def pick_up(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    target_pos_1 = start_pos_1.clone()
    target_pos_1[2] += 0.2

    target_ori_1 = rot_mat_tensor(np.pi, 0, np.pi / 2, env.device)

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]

    return target_ee_states, thresholds, False


def move_to_1(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    part2_pose = env_states[1]
    part1_pose = env_states[0]
    pos, ori = find_part2_pose(env)
    angles = rot_mat_to_angles_tensor(
        torch.tensor(ori, device=part1_pose.device), env.device
    )
    forward = angles[2].item()

    # pos[1] -= 0.1*np.sin(forward)*factor
    # pos[0] += 0.1*np.cos(forward)*factor

    pos_move = torch.tensor(pos, device=part2_pose.device) - part2_pose[:3, 3]

    pos_move[1] += 0.14 * np.cos(forward) * factor
    pos_move[0] -= 0.14 * np.sin(forward) * factor

    target_pos_1 = start_pos_1 + pos_move

    target_pos_1[2] += 0.15

    target_ori_1 = rot_mat_tensor(np.pi, 0, forward - np.pi / 36, env.device)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]

    return target_ee_states, thresholds, False

def move_to_2(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    part2_pose = env_states[1]
    part1_pose = env_states[0]
    pos, ori = find_part2_pose(env)
    angles = rot_mat_to_angles_tensor(
        torch.tensor(ori, device=part1_pose.device), env.device
    )
    forward = angles[2].item()

    # pos[1] -= 0.1*np.sin(forward)*factor
    # pos[0] += 0.1*np.cos(forward)*factor

    pos_move = torch.tensor(pos, device=part2_pose.device) - part2_pose[:3, 3]

    pos_move[1] += 0.14 * np.cos(forward) * factor
    pos_move[0] -= 0.14 * np.sin(forward) * factor

    target_pos_1 = start_pos_1 + pos_move

    # target_pos_1[2] += 0.1

    target_ori_1 = rot_mat_tensor(np.pi, 0, forward, env.device)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]

    return target_ee_states, thresholds, False
    

def move_to_3(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    part2_pose = env_states[1]
    part1_pose = env_states[0]
    pos, ori = find_part2_pose(env)
    angles = rot_mat_to_angles_tensor(
        torch.tensor(ori, device=part1_pose.device), env.device
    )
    forward = angles[2].item()

    # pos[1] -= 0.1*np.sin(forward)*factor
    # pos[0] += 0.1*np.cos(forward)*factor

    pos_move = torch.tensor(pos, device=part2_pose.device) - part2_pose[:3, 3]

    # pos_move[1] += 0.125 * np.cos(forward) * factor
    # pos_move[0] -= 0.125 * np.sin(forward) * factor

    # pos_move[1] -= 0.005 * np.sin(forward) * factor
    # pos_move[0] -= 0.005 * np.cos(forward) * factor

    target_pos_1 = start_pos_1 + pos_move

    # target_pos_1[2] = 0.005

    target_ori_1 = rot_mat_tensor(np.pi, 0, forward, env.device)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]

    return target_ee_states, thresholds, False
# 新增一个专门用于对齐的函数
def align_(env, start_ee_states, env_states):
    # 1. 获取当前的末端位置 (我们只想改变姿态，不想改变位置)
    current_pos, _, gripper_1 = start_ee_states[0]
    
    # 2. 获取目标的、理想的姿态 (以part1为基准)
    part1_pose = env_states[0]
    part1_angles = rot_mat_to_angles_tensor(part1_pose[:3,:3], env.device)
    part1_forward = part1_angles[2].item() # 获取part1的朝向

    # 创造一个与 part1 完全平行的目标姿态
    target_ori_1 = rot_mat_tensor(np.pi, 0, part1_forward, env.device)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)
    
    # 在这里直接修改目标位置的Z值
    target_pos = current_pos.clone()
    target_pos[2] += 0.05

    # 使用新的目标位置和目标姿态
    target_ee_states = [(target_pos, target_quat_1, gripper_1)]
    thresholds = [(0.005, 0.01)]
    
    return target_ee_states, thresholds, False
    

def push_(env, start_ee_states, env_states):
    # 此函数逻辑基本不变，但它现在是在一个完美对齐的姿态下开始的
    # 它只负责计算并移动到最终的插入深度位置
    
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    # 计算最终插入位置 (这里的逻辑可以复用你之前的 insert_ 代码)
    pos, ori = find_part2_pose(env)
    pos_move = torch.tensor(pos, device=env.device) - env_states[1][:3, 3]
    
    # 目标位置是当前位置加上计算出的位移
    # 注意：这里我们不再改变姿态(quat)，因为 align_ 已经把它校准好了
    target_pos_1 = start_pos_1 + pos_move
    
    target_ee_states = [(target_pos_1, start_quat_1, gripper_1)] # 姿态(start_quat_1)保持不变
    thresholds = [(0.005, None)] # 对位置误差设置严格阈值

    return target_ee_states, thresholds, False

# def insert_(env, start_ee_states, env_states):
#     # start_ee_states 是上一步动作结束后的实际位置，它已经包含了误差
#     # env_states 包含 part1 和 part2 的当前精确位置

#     # 1. 计算出理论上完美的插入目标位姿（基于 part1 的当前姿态）
#     # find_part2_pose 函数已经做得很好了，它能得到理想的世界坐标系下的目标位姿
#     ideal_target_pos, ideal_target_ori_matrix = find_part2_pose(env)
#     ideal_target_pos = torch.tensor(ideal_target_pos, device=env.device, dtype=torch.float)
#     ideal_target_ori_matrix = torch.tensor(ideal_target_ori_matrix, device=env.device, dtype=torch.float)
    
#     # 注意：find_part2_pose 返回的是零件的目标位姿，你需要把它转换成末端执行器（EE）的目标位姿。
#     # 这通常需要一个从“被抓取物体”到“EE”的固定变换。这里我们先假设两者位置近似。
#     # (如果抓取点不是物体重心，你需要计算这个偏移)
#     final_target_pos = ideal_target_pos  # 加上从零件到EE的偏移
#     final_target_quat = C.mat2quat(ideal_target_ori_matrix)

#     # 2. 直接将计算出的精确目标作为本次动作的目标，而不是在 start_pos_1 上做累加
#     # 这样做可以消除从 move_to_1 到 move_to_3 累积的所有位置误差。
#     gripper_1 = start_ee_states[0][2] # gripper 状态保持不变
#     target_ee_states = [(final_target_pos, final_target_quat, gripper_1)]
    
#     # 插入时动作要慢且稳，可以增加 pose_spend_time
#     thresholds = [(0.005, 0.01)]  # 可以设置更严格的容忍度阈值 (pos_error, rot_error)
    
#     return target_ee_states, thresholds, False
# def insert_(env, start_ee_states, env_states):
#     factor = env.furniture_scale_factor
#     start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

#     part2_pose = env_states[1]
#     part1_pose = env_states[0]
#     pos, ori = find_part2_pose(env)
#     angles = rot_mat_to_angles_tensor(
#         torch.tensor(ori, device=part2_pose.device), env.device
#     )
#     forward = angles[2].item()
    
#     part1_angles = rot_mat_to_angles_tensor(part1_pose[:3,:3], env.device)
#     part1_forward = part1_angles[2].item()

#     # pos[1] -= 0.1*np.sin(forward)*factor
#     # pos[0] += 0.1*np.cos(forward)*factor

#     pos_move = torch.tensor(pos, device=part2_pose.device) - part2_pose[:3, 3]

#     # pos_move[1] +=0.08*np.cos(forward)*factor
#     # pos_move[0] -=0.08*np.sin(forward)*factor

#     # pos_move[1] -=0.003*np.sin(forward)*factor
#     # pos_move[0] -=0.003*np.cos(forward)*factor

#     target_pos_1 = start_pos_1 + pos_move
#     # target_pos_1[1] +=0.01
#     # target_pos_1[2] += 0.005
#     target_ori_1 = rot_mat_tensor(np.pi, 0, part1_forward, env.device)[:3, :3]
#     target_quat_1 = C.mat2quat(target_ori_1)
#     target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
#     thresholds = [(None, None), (None, None)]

#     return target_ee_states, thresholds, False


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
def pre_push(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item()
    print("Angles: ", part2_angles / np.pi * 180)
    print("Forward: ", part2_forward / np.pi * 180)

    target_ori_1 = rot_mat_tensor(np.pi, 0, part2_forward, env.device)

    target_pos_1 = part2_pos
    # target_pos_1[0] -= 0.037 * np.sin(part2_forward) * factor
    # target_pos_1[1] += 0.037 * np.cos(part2_forward) * factor
    # target_pos_1[0] += 0.01 * np.sin(part2_forward) * factor
    # target_pos_1[1] -= 0.01 * np.cos(part2_forward) * factor
    target_pos_1[2] += 0.012 * factor

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False
    


def pre_push_closer(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]
    part2_pose = env_states[1]
    part2_ori = part2_pose[:3, :3]
    part2_pos = part2_pose[:3, 3]
    part2_angles = rot_mat_to_angles_tensor(part2_ori, env.device)
    part2_forward = part2_angles[2].item()
    print("Angles: ", part2_angles / np.pi * 180)
    print("Forward: ", part2_forward / np.pi * 180)
    target_ori_1 = rot_mat_tensor(np.pi-np.pi/18, 0, part2_forward, env.device)

    target_pos_1 = part2_pos
    # target_pos_1[0] -= 0.037 * np.sin(part2_forward) * factor
    # target_pos_1[1] += 0.037 * np.cos(part2_forward) * factor
    # target_pos_1[0] += 0.012 * np.sin(part2_forward) * factor
    # target_pos_1[1] -= 0.012 * np.cos(part2_forward) * factor
    target_pos_1[2] += 0.012 * factor

    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, None), (None, None)]
    return target_ee_states, thresholds, False


def push(env, start_ee_states, env_states):
    factor = env.furniture_scale_factor
    start_pos_1, start_quat_1, gripper_1 = start_ee_states[0]

    part2_pose = env_states[1]
    part1_pose = env_states[0]
    pos, ori = find_part2_pose(env)
    angles = rot_mat_to_angles_tensor(
        torch.tensor(ori, device=part2_pose.device), env.device
    )
    forward = angles[2].item()
    
    part1_angles = rot_mat_to_angles_tensor(part1_pose[:3,:3], env.device)
    part1_forward = part1_angles[2].item()

    # pos[1] -= 0.1*np.sin(forward)*factor
    # pos[0] += 0.1*np.cos(forward)*factor

    pos_move = torch.tensor(pos, device=part2_pose.device) - part2_pose[:3, 3]

    # pos_move[1] +=0.08*np.cos(forward)*factor
    # pos_move[0] -=0.08*np.sin(forward)*factor

    # pos_move[1] -=0.003*np.sin(forward)*factor
    # pos_move[0] -=0.003*np.cos(forward)*factor

    target_pos_1 = start_pos_1 + pos_move
    # target_pos_1[1] +=0.01
    target_pos_1[2] += 0.005
    target_ori_1 = rot_mat_tensor(np.pi-np.pi/18, 0, part1_forward, env.device)[:3, :3]
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
    "pick_": pick_,
    "pick_closer": pick_closer,
    "pick_up": pick_up,
    "move_to_1": move_to_1,
    "move_to_2": move_to_2,
    "move_to_3": move_to_3,
    "align_": align_,
    "push_": push_,
    # "insert_": insert_,
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
    # print("part1_name: ", part1_name)

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
        pose_spend_time = 30

    result = reach_target(
        env,
        target_ee_states,
        thresholds,
        is_gripper,
        slow=slow,
        pose_spend_time=pose_spend_time,
    )

    return target_ee_states, result

def pick_prepare(env, prev_target_ee_states):
    target_ee_states, result = act_phase(env, "pick_", func_map, prev_target_ee_states)
    target_ee_states, result = act_phase(env, "pick_closer", func_map, target_ee_states)
    target_ee_states, result = act_phase(env, "close_gripper", func_map, target_ee_states)
    target_ee_states, result = act_phase(env, "pick_up", func_map, target_ee_states)
    return target_ee_states, result

def pick_perform(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    target_ee_states, result = act_phase(env, "move_to_1", func_map, target_ee_states)
    target_ee_states, result = act_phase(env, "move_to_2", func_map, target_ee_states,)
    target_ee_states, result = act_phase(env, "move_to_3", func_map, target_ee_states)
    
    target_ee_states, result = act_phase(env, "align_", func_map, target_ee_states, slow=True) 
    target_ee_states, result = act_phase(env, "push_", func_map, target_ee_states, slow=True)
    
    # target_ee_states, result = act_phase(env, "insert_", func_map, target_ee_states, slow=True)
    target_ee_states, result = act_phase(env, "release_gripper", func_map, target_ee_states)
    return target_ee_states, result

def prepare(env, prev_target_ee_states):
    # target_ee_states, result = act_phase(env, "pick_", func_map, prev_target_ee_states)
    # target_ee_states, result = act_phase(env, "pick_closer", func_map, target_ee_states)
    # target_ee_states, result = act_phase(env, "close_gripper", func_map, target_ee_states)
    # target_ee_states, result = act_phase(env, "pick_up", func_map, target_ee_states)
    # target_ee_states, result = act_phase(env, "move_to_1", func_map, target_ee_states)
    # target_ee_states, result = act_phase(env, "move_to_2", func_map, target_ee_states,slow=True)
    # target_ee_states, result = act_phase(env, "insert_", func_map, target_ee_states,slow=True)
    target_ee_states, result = act_phase(env, "pre_push", func_map, prev_target_ee_states)
    target_ee_states, result = act_phase(env, "pre_push_closer", func_map, target_ee_states)

    return target_ee_states, result


def perform(env, prev_target_ee_states):

    target_ee_states = prev_target_ee_states
    # target_ee_states, result = act_phase(env, "close_gripper", func_map, target_ee_states)
    # target_ee_states, result = act_phase(env, "close_gripper", func_map, target_ee_states)
   
    target_ee_states, result = act_phase(env, "push", func_map, target_ee_states)
    target_ee_states, result = act_phase(env, "push", func_map, target_ee_states)
    # target_ee_states, result = act_phase(env, "insert_", func_map, target_ee_states)
    # target_ee_states, result = act_phase(env, "insert_", func_map, target_ee_states, slow=True)

    return True


def perform_disassemble(env, prev_target_ee_states):
    for i in trange(2):
        wait(env)
    return
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
