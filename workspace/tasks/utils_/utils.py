import torch
from workspace.tasks.utils import reach_target
import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T


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


def act_phase(
    env, phase, func_map, part1_name, part2_name, last_target_ee_states=None, slow=False
):
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
        gripper_1 = env.last_grasp

    ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
    gripper_1 = gripper_1.squeeze()

    start_ee_states = [(ee_pos_1, ee_quat_1, gripper_1)]
    env_states = [part1_pose, part2_pose]

    target_ee_states, thresholds, is_gripper = func_map[phase](
        env, start_ee_states, env_states
    )

    result = reach_target(env, target_ee_states, thresholds, is_gripper, slow=slow)

    return target_ee_states, result


def get_assemled_pose(env, part1_name, part2_name):
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
