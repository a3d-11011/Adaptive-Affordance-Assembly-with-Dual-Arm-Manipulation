from isaacgym import gymapi, gymtorch
import torch
import numpy as np
import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from workspace.utils.rotation import is_tipped
from workspace.utils.distance import (
    double_geodesic_distance_between_poses,
    geodesic_distance_between_R,
)
from workspace.utils.direction import sample_dir, dir2pos
from workspace.utils.point_cloud import (
    obs_to_pc_fps,
    find_farthest_nearby_point_torch,
    find_four_by_proj_and_perp,
)
from workspace.utils.trajectory import dense_trajectory_points_generation
import open3d as o3d


# Matrix Operations
def rot_mat(angles, hom: bool = False):
    """Given @angles (x, y, z), compute rotation matrix
    Args:
        angles: (x, y, z) rotation angles.
        hom: whether to return a homogeneous matrix.
    """
    x, y, z = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    if hom:
        M = np.zeros((4, 4), dtype=np.float32)
        M[:3, :3] = R
        M[3, 3] = 1.0
        return M
    return R


def rot_mat_to_angles(R):
    """
    Given a 3x3 rotation matrix, compute the rotation angles (x, y, z).
    Args:
        R: 3x3 rotation matrix.
    Returns:
        angles: (x, y, z) rotation angles.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-4

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rot_mat_to_angles_tensor(R, device):
    """
    Given a 3x3 rotation matrix, compute the rotation angles (x, y, z).
    Args:
        R: 3x3 rotation matrix.
    Returns:
        angles: (x, y, z) rotation angles.
    """
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-4

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z], device=device)


def rot_mat_to_angles_tensor_1(R: torch.Tensor, device):
    """
    3x3 旋转矩阵 -> ZYX 顺序 Euler 角 (roll, pitch, yaw)
    R: [..., 3, 3]
    返回 angles: [..., 3]，顺序是 (roll_x, pitch_y, yaw_z)
    """
    eps = 1e-6
    # 把 R[2,0] 钳到 [-1,1]
    r20 = R[..., 2, 0].clamp(-1.0, 1.0)
    # 是否万向锁
    singular = r20.abs() > (1.0 - eps)

    # non-singular 分支
    def normal_branch():
        # pitch = -asin(r20)
        pitch = -torch.asin(r20)
        cos_pitch = torch.cos(pitch)
        # roll = atan2(R[2,1]/cos, R[2,2]/cos)
        roll = torch.atan2(R[..., 2, 1] / cos_pitch, R[..., 2, 2] / cos_pitch)
        # yaw  = atan2(R[1,0]/cos, R[0,0]/cos)
        yaw = torch.atan2(R[..., 1, 0] / cos_pitch, R[..., 0, 0] / cos_pitch)
        return roll, pitch, yaw

    # singular 分支（gimbal-lock）
    def singular_branch():
        # pitch = -sign(r20)*π/2
        pitch = -r20.sign() * (torch.pi / 2)
        # 强制 roll = 0
        roll = torch.zeros_like(pitch)
        # yaw = atan2(-R[0,1], R[1,1])  （或你习惯的另一个不含 roll 的公式）
        yaw = torch.atan2(-R[..., 0, 1], R[..., 1, 1])
        return roll, pitch, yaw

    # 根据条件挑分支
    roll, pitch, yaw = torch.where(
        singular.unsqueeze(-1),
        torch.stack(singular_branch(), dim=-1),
        torch.stack(normal_branch(), dim=-1),
    ).unbind(-1)

    # 最终合并成 (...,3)
    return torch.stack([roll, pitch, yaw], dim=-1).to(device)


def rot_mat_tensor(x, y, z, device):
    return torch.tensor(rot_mat([x, y, z], hom=True), device=device).float()


def rel_rot_mat(s, t):
    s_inv = torch.linalg.inv(s)
    return t @ s_inv


def satisfy(
    current,
    target,
    pos_error_threshold=None,
    ori_error_threshold=None,
) -> bool:
    default_pos_error_threshold = 0.01
    default_ori_error_threshold = 0.2

    if pos_error_threshold is None:
        pos_error_threshold = default_pos_error_threshold
    if ori_error_threshold is None:
        ori_error_threshold = default_ori_error_threshold

    if ((current[:3, 3] - target[:3, 3]).abs().sum() < pos_error_threshold) and (
        (target[:3, :3] - current[:3, :3]).abs().sum() < ori_error_threshold
    ):
        return True
    return False


def gripper_less(gripper_width, target_width):
    if gripper_width <= target_width:
        return True
    return False


def small_wait(env):
    for i in range(10):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # action = torch.cat((action, new_action), dim=1)

        env.step(action)


def wait(env):
    for i in range(30):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # action = torch.cat((action, new_action), dim=1)
        env.step(action)


def wait_close(env, times=30):
    for i in range(times):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], device=env.device).unsqueeze(0)
        # new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # action = torch.cat((action, new_action), dim=1)
        env.step(action)


# Control Operations
def get_action(
    env, start_pos, start_quat, target_pos, target_quat, gripper, slow=False
):
    delta_pos = target_pos - start_pos

    # Scale translational action.
    delta_pos_sign = delta_pos.sign()
    delta_pos = torch.abs(delta_pos) * 2
    for i in range(3):
        if delta_pos[i] > 0.03:
            delta_pos[i] = 0.03 + (delta_pos[i] - 0.03) * np.random.normal(1.5, 0.1)
    delta_pos = delta_pos * delta_pos_sign

    # Clamp too large action.
    max_delta_pos = 0.10 + 0.01 * torch.rand(3, device=env.device)
    max_delta_pos[2] -= 0.04
    delta_pos = torch.clamp(delta_pos, min=-max_delta_pos, max=max_delta_pos)
    if slow:
        delta_pos = delta_pos / 2
    target_quat = adj_quat(start_quat, target_quat)
    delta_quat = C.quat_mul(C.quat_conjugate(start_quat), target_quat)

    gripper = torch.tensor([gripper], device=env.device)
    action = torch.concat([delta_pos, delta_quat, gripper]).unsqueeze(0)
    return action


def adj_quat(q0, q1):
    if torch.dot(q0, q1) < 0:
        q1 = -q1
    return q1


def reach_target(
    env,
    target_ee_states,
    thresholds,
    is_gripper,
    gripper_spend_time=10,
    pose_spend_time=50,
    slow=False,
):
    if hasattr(env, "need_reset") and env.need_reset is True:
        return False
    target_pos_1, target_quat_1, gripper_1 = target_ee_states[0]
    pos_err_1, ori_err_1 = thresholds[0]

    spend_time = 0

    while True:
        ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
        ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
        action_1 = get_action(
            env, ee_pos_1, ee_quat_1, target_pos_1, target_quat_1, gripper_1, slow=slow
        )
        action = action_1

        ee_pose_1 = C.to_homogeneous(ee_pos_1, C.quat2mat(ee_quat_1))
        target_pose_1 = C.to_homogeneous(target_pos_1, C.quat2mat(target_quat_1))

        gripper_width = 1
        half_width = 0

        if is_gripper:
            if (
                gripper_less(gripper_width, 2 * half_width + 0.001)
                or spend_time > gripper_spend_time
            ):
                return True
        else:
            if satisfy(ee_pose_1, target_pose_1, pos_err_1, ori_err_1) and (
                not hasattr(env, "smooth_hand_p")
                or env.smooth_hand_p is None
                or len(env.smooth_hand_p) == 0
            ):  # and satisfy(ee_pose_2, target_pose_2, pos_err_2, ori_err_2, spend_time=spend_time):
                return True
            if spend_time > pose_spend_time:
                print("Time out, not reach target.")
                return False

        if (
            hasattr(env, "smooth_hand_p")
            and env.smooth_hand_p is not None
            and len(env.smooth_hand_p) > 0
        ):
            p_i = env.smooth_hand_p[0]
            q_i = env.smooth_hand_q[0]
            m_i = T.quat2mat(q_i)
            env.smooth_hand_p = env.smooth_hand_p[1:]
            env.smooth_hand_q = env.smooth_hand_q[1:]
            env.set_hand_transform(p_i, m_i)

        env.step(action)
        # if reset_hand(env) is False:
        #     return False

        if is_moved_check(env):
            if hasattr(env, "smooth_hand_p"):
                env.smooth_hand_p = None
                env.smooth_hand_q = None
            return False

        spend_time += 1


def is_moved_check(env, time_gap=8, threshold=0.20):
    if "bucket" in env.task_config["task_name"] or "drawer_c" in env.task_config["task_name"] or "cask" in env.task_config["task_name"]:
        time_gap += 16
        threshold = 0.095
    elif ("drawer" in env.task_config["task_name"]) or (
        "cabinet" in env.task_config["task_name"]
    ):
        threshold -= 0.085
    elif "pull" in env.task_config["task_name"]:
        threshold -= 0.08
    # if "desk" in env.task_config["task_name"] and "pull" in env.task_config["task_name"]:
    #     threshold =0.20
    
    if env.reset_check is False:
        return False
    if env.need_reset is True:
        return True
    # if env.reset_failed is True:
    #     return False
    reset_times = len(env.sampled_points) - 1
    if reset_times >= env.reset_t:
        return False

    cur_step = len(env.observation[0]) - 1

    if cur_step - env.last_check_step < time_gap:
        return False

    cur_obs = env.observation[0][-1]
    pre_obs = env.observation[0][env.last_check_step]
    cur_pose = cur_obs["part_pose"][env.part1_name]
    pre_pose = pre_obs["part_pose"][env.part1_name]

    cur_pose = C.to_homogeneous(
        cur_pose[:3],
        C.quat2mat(cur_pose[3:7]),
    )
    pre_pose = C.to_homogeneous(
        pre_pose[:3],
        C.quat2mat(pre_pose[3:7]),
    )
    distance = double_geodesic_distance_between_poses(cur_pose, pre_pose)
    if "bucket" in env.task_config["task_name"] or "drawer_c" in env.task_config["task_name"] or "cask" in env.task_config["task_name"]:
        distance = geodesic_distance_between_R(cur_pose[:3, :3], pre_pose[:3, :3])
    print(f"Cur:{cur_step},Pre:{env.env.last_check_step},Distance:{distance}")

    if distance > threshold:
        env.moved.append(env.last_check_step)
        print("Part moved")
        env.need_reset = True
        return True
    elif "bucket" in env.task_config["task_name"] or "cask" in env.task_config["task_name"]:

        tipped, deg = is_tipped(cur_pose[:3, :3], 5.0)
        if tipped:
            env.moved.append(env.last_check_step)
            print(f"Part tipped:{deg} degrees")
            env.need_reset = True
            return True
    elif "drawer_c" in env.task_config["task_name"]:
        tipped, deg = is_tipped(cur_pose[:3, :3], 6.0,1,True)
        if tipped:
            env.moved.append(env.last_check_step)
            print(f"Part tipped:{deg} degrees")
            env.need_reset = True
            return True

    env.set_last_check_step(cur_step)
    return False


def reach_target(
    env,
    target_ee_states,
    thresholds,
    is_gripper,
    gripper_spend_time=10,
    pose_spend_time=50,
    slow=False,
):
    if hasattr(env, "need_reset") and env.need_reset is True:
        return False
    target_pos_1, target_quat_1, gripper_1 = target_ee_states[0]
    pos_err_1, ori_err_1 = thresholds[0]

    spend_time = 0

    while True:
        ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
        ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
        action_1 = get_action(
            env, ee_pos_1, ee_quat_1, target_pos_1, target_quat_1, gripper_1, slow=slow
        )
        action = action_1

        ee_pose_1 = C.to_homogeneous(ee_pos_1, C.quat2mat(ee_quat_1))
        target_pose_1 = C.to_homogeneous(target_pos_1, C.quat2mat(target_quat_1))

        gripper_width = 1
        half_width = 0

        if is_gripper:
            if (
                gripper_less(gripper_width, 2 * half_width + 0.001)
                or spend_time > gripper_spend_time
            ):
                return True
        else:
            if satisfy(ee_pose_1, target_pose_1, pos_err_1, ori_err_1) and (
                not hasattr(env, "smooth_hand_p")
                or env.smooth_hand_p is None
                or len(env.smooth_hand_p) == 0
            ):  # and satisfy(ee_pose_2, target_pose_2, pos_err_2, ori_err_2, spend_time=spend_time):
                return True
            if spend_time > pose_spend_time:
                print("Time out, not reach target.")
                return False

        if (
            hasattr(env, "smooth_hand_p")
            and env.smooth_hand_p is not None
            and len(env.smooth_hand_p) > 0
        ):
            p_i = env.smooth_hand_p[0]
            q_i = env.smooth_hand_q[0]
            m_i = T.quat2mat(q_i)
            env.smooth_hand_p = env.smooth_hand_p[1:]
            env.smooth_hand_q = env.smooth_hand_q[1:]
            env.set_hand_transform(p_i, m_i)

        env.step(action)
        # if reset_hand(env) is False:
        #     return False

        if is_moved_check(env):
            if hasattr(env, "smooth_hand_p"):
                env.smooth_hand_p = None
                env.smooth_hand_q = None
            return False

        spend_time += 1


def heuristic_point_sample(points, cur_pose, task="drawer", threshold=0.10, show=False):
    rev = task.split("_")[-1] == "r"
    if cur_pose.shape[0] == 7:
        part1_pose = C.to_homogeneous(
                cur_pose[:3],
                C.quat2mat(cur_pose[3:7]),
            )
    else:
        part1_pose = cur_pose
    if "desk" in task or "square_table" in task or "lamp" in task or "chair" in task:
        
        part1_angle = rot_mat_to_angles_tensor(part1_pose[:3, :3], part1_pose.device)
        part1_forward = part1_angle[2].item()
        part1_forward +=np.pi/2
        part1_backward = torch.tensor(
            [np.sin(part1_forward), -np.cos(part1_forward), 0.0],
            dtype=torch.float32,
            device=part1_pose.device,
        )
        part1_forward = -part1_backward
        part1_pos = part1_pose[:3, 3]

        dis = 0.06
        if "lamp" in task or "chair" in task:
            dis = 0.03

        ids, sample_points = find_four_by_proj_and_perp(
            points, part1_pos, part1_backward, dis
        )
        if rev:
            # sampled_point = sample_points[2]
            valid_indices = ids[1]
        else:
            # sampled_point = sample_points[3]
            valid_indices = ids[2]
    elif "drawer_c" in task:
        # threshold = 0.9
        # part1_angles = rot_mat_to_angles_tensor(part1_pose[:3,:3],device=points.device)
        # part1_forward = part1_angles[2].item()+np.pi
        # forward_normal = torch.tensor([np.cos(part1_forward), -np.sin(part1_forward), 0.0], dtype=torch.float32, device=points.device)
        # d_i=points - cur_pose[:3]
        # proj_=torch.sum(d_i*forward_normal,dim=1)
        # valid_indices=torch.nonzero(proj_>threshold).squeeze()
        threshold = 0.9
        z_rel = points[:, 2] - part1_pose[2,3]
        z_min, z_max = z_rel.min(), z_rel.max()
        z_norm = (z_rel - z_min) / (z_max - z_min + 1e-6)
        valid_indices = torch.nonzero(z_norm > threshold).squeeze()
    elif "drawer" in task:
        threshold=0.09
        part1_angles = rot_mat_to_angles_tensor(
            part1_pose[:3, :3], device=points.device
        )
        part1_forward = part1_angles[2].item() + np.pi
        if rev:
            threshold =0.12
            part1_forward -= np.pi
        forward_normal = torch.tensor(
            [-np.sin(part1_forward), np.cos(part1_forward), 0.0],
            dtype=torch.float32,
            device=points.device,
        )
        d_i = points - part1_pose[:3,3]
        proj_ = torch.sum(d_i * forward_normal, dim=1)
        valid_indices = torch.nonzero(proj_ > threshold).squeeze()

    elif "bucket" in task:
        # threshold = 0.16
        # part1_angles = rot_mat_to_angles_tensor(part1_pose[:3,:3],device=points.device)
        # part1_forward = part1_angles[2].item()+np.pi
        # forward_normal = torch.tensor([-np.sin(part1_forward), np.cos(part1_forward), 0.0], dtype=torch.float32, device=points.device)
        # d_i=points - part1_pose[:3,3]
        # proj_=torch.sum(d_i*forward_normal,dim=1)
        # valid_indices=torch.nonzero(proj_>threshold).squeeze()
        threshold = 0.9
        z_rel = points[:, 2] - part1_pose[2,3]
        z_min, z_max = z_rel.min(), z_rel.max()
        z_norm = (z_rel - z_min) / (z_max - z_min + 1e-6)
        valid_indices = torch.nonzero(z_norm > threshold).squeeze()

    valid_indices = valid_indices.view(-1)
    if valid_indices.numel() == 0:
        print("No valid point")
        random_index = torch.randint(
            0, points.shape[0], (1,), device=points.device
        ).item()
    else:
        idx = torch.randint(0, valid_indices.numel(), (), device=points.device).item()
        random_index = valid_indices[idx]
    if show:
        pcs = points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs.cpu().numpy())
        color = np.zeros_like(pcd.points)
        color[:, 2] = 1.0  # Red for all points
        color[valid_indices.cpu().numpy()] = [0, 1, 0]
        color[random_index.cpu().numpy()] = [1, 0, 0]

        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])
    return random_index


def reset_hand_(
    env, show=False, insert=False, one_part=False, conditioned=False, vertical=False
):
    cur_step = len(env.observation[0]) - 1
    cur_obs = env.observation[0][-1]
    pre_obs = env.observation[0][env.last_check_step]
    cur_pose = cur_obs["part_pose"][env.part1_name]
    pre_pose = pre_obs["part_pose"][env.part1_name]
    if not one_part:
        result = obs_to_pc_fps(cur_obs, env.camera_matix[0], env.img_size, [1, 2], 2048)
        if result is False:
            print("Part out of view")
            return False
        points, normals = result
        if show:
            pcs = torch.cat((points[1], points[2]), dim=0)[:, :3]
            # pcs=points[2][:,:3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcs.cpu().numpy())
            o3d.visualization.draw_geometries([pcd])
    else:
        result = obs_to_pc_fps(cur_obs, env.camera_matix[0], env.img_size, [1], 2048)
        if result is False:
            print("Part out of view")
            return False
        points, normals = result
        if show:
            pcs = points[1][:, :3]
            # pcs=points[2][:,:3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcs.cpu().numpy())
            o3d.visualization.draw_geometries([pcd])

    points = points[1]
    normals = normals[1]

    for i in range(1000):
        if conditioned:
            random_index = heuristic_point_sample(
                points, cur_pose, env.task_config["task_name"], show=show
            )
        else:
            random_index = torch.randint(0, points.shape[0], (1,), device=points.device)
        re_sampled_point = points[random_index].squeeze(0)
        re_sampled_normal = normals[random_index].squeeze(0)
        result = sample_dir(re_sampled_point, re_sampled_normal, vertical=vertical)
        if result is False:
            print("Resample Hand Pose Collision")
            continue
        else:
            hand_pos, hand_ori, action_direct = result
            try:
                env.set_hand_transform(hand_pos, hand_ori)
                if insert:
                    hand_pos -= 0.01 * action_direct.cpu().numpy()
                    smooth_hand_transform(env, hand_pos, hand_ori)
            except:
                print("Hand pos out of bound:", hand_pos, hand_ori)
                continue
            contact_flag = env.check_contact()
            if contact_flag is False:
                env.moved.append(cur_step)
                env.sampled_points.append(re_sampled_point)
                env.sampled_normals.append(re_sampled_normal)
                env.action_directs.append(action_direct)
                return True
            else:
                print("Hand pos collision")
                # env.reset_failed=True
                return False


def reset_hand(env, time_gap=8, reset_t=1, threshold=0.15, show=False):
    if (
        ("drawer" in env.task_config["task_name"])
        or ("cabinet" in env.task_config["task_name"])
        or ("pull" in env.task_config["task_name"])
    ):
        threshold -= 0.085
    if env.reset_check is False:
        return True
    if env.reset_failed is True:
        return False
    reset_times = len(env.sampled_points) - 1
    if reset_times >= reset_t:
        return True

    cur_step = len(env.observation[0]) - 1

    if cur_step - env.last_check_step < time_gap:
        return True

    cur_obs = env.observation[0][-1]
    pre_obs = env.observation[0][env.last_check_step]
    cur_pose = cur_obs["part_pose"][env.part1_name]
    pre_pose = pre_obs["part_pose"][env.part1_name]

    cur_pose = C.to_homogeneous(
        cur_pose[:3],
        C.quat2mat(cur_pose[3:7]),
    )
    pre_pose = C.to_homogeneous(
        pre_pose[:3],
        C.quat2mat(pre_pose[3:7]),
    )
    distance = double_geodesic_distance_between_poses(cur_pose, pre_pose)

    print(f"Cur:{cur_step},Pre:{env.env.last_check_step},Distance:{distance}")
    env.set_last_check_step(cur_step)
    if distance > threshold:
        print("Part moved")
        result = obs_to_pc_fps(cur_obs, env.camera_matix[0], env.img_size, [1, 2], 2048)
        if result is False:
            print("Part out of view")
            return False
        points, normals = result
        if show:
            pcs = torch.cat((points[1], points[2]), dim=0)[:, :3]
            # pcs=points[2][:,:3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcs.cpu().numpy())
            o3d.visualization.draw_geometries([pcd])
        points = points[1]
        normals = normals[1]

        for i in range(1000):
            random_index = torch.randint(0, points.shape[0], (1,), device=points.device)
            re_sampled_point = points[random_index].squeeze(0)
            re_sampled_normal = normals[random_index].squeeze(0)
            result = sample_dir(re_sampled_point, re_sampled_normal)
            if result is False:
                print("Resample Hand Pose Collision")
                continue
            else:
                hand_pos, hand_ori, action_direct = result
                try:
                    env.set_hand_transform(hand_pos, hand_ori)
                except:
                    print("Hand pos out of bound:", hand_pos, hand_ori)
                    continue
                contact_flag = env.check_contact()
                if contact_flag is False:
                    env.moved.append(cur_step)
                    env.sampled_points.append(re_sampled_point)
                    env.sampled_normals.append(re_sampled_normal)
                    env.action_directs.append(action_direct)
                    break
                else:
                    print("Hand pos collision")
                    env.reset_failed = True
                    return False


def move_franka_away(env, distance=0.1):
    thresholds = [(None, None)]
    ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
    ee_pos_1, ee_quat_1 = ee_pos_1[0], ee_quat_1[0]
    target_ee_states = [(ee_pos_1, ee_quat_1, -1)]
    reach_target(env, target_ee_states, thresholds, True)
    R = C.quat2mat(ee_quat_1).to(device=env.device)
    opposite_direction = R[:, 2]
    opposite_direction = -opposite_direction / opposite_direction.norm()
    target_pos1 = ee_pos_1 + opposite_direction * distance
    target_ee_states = [(target_pos1, ee_quat_1, -1)]
    reach_target(env, target_ee_states, thresholds, False)
    target_ee_states = [
        (
            torch.tensor([0.4, 0.445, 0.7], device=env.device),
            torch.tensor([0.9, 0.45, 0.0, 0.0], device=env.device),
            -1,
        )
    ]

    reach_target(env, target_ee_states, thresholds, False)


def sample_hand_pose(network, pcs, cp1, interaction, rvs=10):
    aff = network["aff"]
    actor = network["actor"]
    critic = network["critic"]
    device = network["device"]

    pcs = pcs.unsqueeze(0).to(device)
    cp1 = cp1.unsqueeze(0).to(device)
    interaction = [i.unsqueeze(0).to(device) for i in interaction]

    score = aff.inference_whole_pc(pcs, cp1, interaction).squeeze(0)[:2048]

    max_index = torch.argmax(score)
    max_index = max_index.item()
    sampled_point = pcs[0, max_index].unsqueeze(0)
    recon_dir2 = (
        actor.actor_sample_n(pcs, cp1, sampled_point, interaction, rvs)
        .contiguous()
        .view(rvs, -1)
    )
    action_score = critic.forward_n(
        pcs, cp1, sampled_point, recon_dir2, interaction, rvs
    )
    action_score = action_score.contiguous().view(rvs, 1)
    ac_index = torch.argmax(action_score).item()
    action_direct = recon_dir2[ac_index]

    hand_pos, hand_ori = dir2pos(sampled_point.squeeze(0)[:3], action_direct)

    return hand_pos, hand_ori, sampled_point, action_direct


def sample_hand_pose_heuristic(env, pcs):
    rb_states = env.rb_states
    part_idxs = env.part_idxs
    part1_name = env.part1_name
    part2_name = env.part2_name
    part1_pose = C.to_homogeneous(
        rb_states[part_idxs[part1_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
    )
    part2_pose = C.to_homogeneous(
        rb_states[part_idxs[part2_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
    )

    part1_angle = rot_mat_to_angles_tensor(part1_pose[:3, :3], part1_pose.device)
    part1_forward = part1_angle[2].item()
    part1_backward = torch.tensor(
        [np.sin(part1_forward), -np.cos(part1_forward), 0.0],
        dtype=torch.float32,
        device=part1_pose.device,
    )
    part1_forward = -part1_backward
    part1_pos = part1_pose[:3, 3]

    if (
        "desk" in env.task_config["task_name"]
        or "square_table" in env.task_config["task_name"]
        or "lamp" in env.task_config["task_name"]
        or "chair" in env.task_config["task_name"]
    ):

        dis = 0.06
        if (
            "lamp" in env.task_config["task_name"]
            or "chair" in env.task_config["task_name"]
        ):
            dis = 0.03
        dis *= env.furniture_scale_factor
        _, sample_points = find_four_by_proj_and_perp(
            pcs, part1_pos, part1_backward, dis
        )
        if env.task_config["task_name"].split("_")[-1] != "r":
            sampled_point = sample_points[2]
        else:
            sampled_point = sample_points[3]
    elif (
        "drawer" in env.task_config["task_name"]
        or "cabinet" in env.task_config["task_name"]
    ):
        _, sampled_point = find_farthest_nearby_point_torch(
            pcs, part1_pos, part1_backward, d_max=0.05
        )
        if sampled_point is None:
            print("No sampled point")
            return None, None, None, None
    else:
        raise ("Not define task heuristic!")

    hand_pos, hand_ori, action_direct = sample_dir(
        sampled_point[0:3], sampled_point[3:6]
    )
    return hand_pos, hand_ori, sampled_point, action_direct


def set_hand_transform_with_rt(env, target_pos, target_ori, max_step=None):
    p = env.franka_hand_pose.p
    r = env.franka_hand_pose.r
    cur_pos = np.array([p.x, p.y, p.z])
    cur_q = np.array([r.x, r.y, r.z, r.w])
    if target_ori.shape == (3,):
        target_q = T.euler2quat(target_ori)
    else:
        target_q = target_ori

    p_t, q_t = dense_trajectory_points_generation(cur_pos, target_pos, cur_q, target_q)
    env.smooth_hand_p = p_t
    env.smooth_hand_q = q_t


def smooth_hand_transform(env, target_pos, target_ori, max_step=None):
    p = env.franka_hand_pose.p
    r = env.franka_hand_pose.r
    cur_pos = np.array([p.x, p.y, p.z])
    cur_q = np.array([r.x, r.y, r.z, r.w])
    if target_ori.shape == (3,):
        target_q = T.euler2quat(target_ori)
    elif target_ori.shape == (4,):
        target_q = target_ori
    else:
        # elif target_ori.shape == (3,3):
        target_q = T.mat2quat(target_ori[:3, :3])
    # else:
    #     print(target_ori.shape)

    p_t, q_t = dense_trajectory_points_generation(cur_pos, target_pos, cur_q, target_q)

    for i in range(len(p_t)):
        pos = p_t[i]
        q = q_t[i]
        m = T.quat2mat(q)
        env.set_hand_transform(pos, m)
        env.step(
            torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        )


def control_hand(env, close=True, max_step=10):
    """
    Close or Release the fly gripper
    Only used for the finger movable franka hand
    """
    env.close_hand(close)
    for i in range(max_step):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        env.step(action)


# Point Cloud Operations
def depth_image_to_point_cloud_GPU(
    depth_image,
    camera_view_matrix,
    camera_proj_matrix,
    width: float,
    height: float,
    depth_bar: float = None,
    device: torch.device = "cuda:0",
):
    vinv = torch.inverse(camera_view_matrix)
    proj = camera_proj_matrix
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    u = torch.linspace(0, width - 1, width, device=device)
    v = torch.linspace(0, height - 1, height, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    Z = depth_image
    x_para = -Z * fu / width
    y_para = Z * fv / height
    X = (u - centerU) * x_para
    Y = (v - centerV) * y_para

    # valid = Z > -0.8
    position = torch.stack([X, Y, Z, torch.ones_like(X)], dim=-1)
    position = position.view(-1, 4)
    position = position @ vinv
    points = position[:, :3]

    # normal_image = normal_image.permute(1, 2, 0).view(-1, 3)  # 将法线图像转换为 (N, 3) 形状
    # normal_image = normal_image @ vinv[:3, :3]

    x_dz = (
        depth_image[1 : height - 1, 2:width]
        - depth_image[1 : height - 1, 0 : width - 2]
    ) * 0.5
    y_dz = (
        depth_image[2:height, 1 : width - 1]
        - depth_image[0 : height - 2, 1 : width - 1]
    ) * 0.5
    dx = x_para
    dy = y_para
    dx = dx[1 : height - 1, 1 : width - 1]
    dy = dy[1 : height - 1, 1 : width - 1]

    normal_x = -x_dz / dx
    normal_y = -y_dz / dy
    normal_z = torch.ones((height - 2, width - 2), device=device)

    normal_l = torch.sqrt(
        normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
    )
    normal_x = normal_x / normal_l
    normal_y = normal_y / normal_l
    normal_z = normal_z / normal_l

    normal_map = torch.stack([normal_x, normal_y, normal_z], dim=-1)
    normal_map_full = torch.zeros((height, width, 3)).to(depth_image.device)
    normal_map_full[1 : height - 1, 1 : width - 1, :] = normal_map
    normals = normal_map_full.view(-1, 3)
    normals = normals @ vinv[:3, :3]

    return points, normals


def capture_pc(env, camera_handle):
    # camera_handle = env.camera_handles["front"][0]
    cam_proj = torch.tensor(
        env.isaac_gym.get_camera_proj_matrix(env.sim, env.envs[0], camera_handle)
    ).to(env.device)
    cam_view = torch.tensor(
        env.isaac_gym.get_camera_view_matrix(env.sim, env.envs[0], camera_handle)
    ).to(env.device)

    depth_render_type = gymapi.IMAGE_DEPTH
    depth_image = gymtorch.wrap_tensor(
        env.isaac_gym.get_camera_image_gpu_tensor(
            env.sim, env.envs[0], camera_handle, depth_render_type
        )
    )

    points, normals = depth_image_to_point_cloud_GPU(
        depth_image,
        cam_view,
        cam_proj,
        width=env.img_size[0],
        height=env.img_size[1],
    )

    seg_render_type = gymapi.IMAGE_SEGMENTATION
    seg_image = gymtorch.wrap_tensor(
        env.isaac_gym.get_camera_image_gpu_tensor(
            env.sim, env.envs[0], camera_handle, seg_render_type
        )
    )
    segments = seg_image.flatten()

    return points, normals, segments


def capture_pc_headless(env, camera_handle):
    cam_proj = torch.tensor(
        env.isaac_gym.get_camera_proj_matrix(env.sim, env.envs[0], camera_handle)
    ).to(env.device)
    cam_view = torch.tensor(
        env.isaac_gym.get_camera_view_matrix(env.sim, env.envs[0], camera_handle)
    ).to(env.device)

    depth_render_type = gymapi.IMAGE_DEPTH
    depth_image = torch.tensor(
        env.isaac_gym.get_camera_image(
            env.sim, env.envs[0], camera_handle, depth_render_type
        )
    ).to(env.device)

    points, normals = depth_image_to_point_cloud_GPU(
        depth_image,
        cam_view,
        cam_proj,
        width=env.img_size[0],
        height=env.img_size[1],
    )

    seg_render_type = gymapi.IMAGE_SEGMENTATION
    seg_image = torch.tensor(
        env.isaac_gym.get_camera_image(
            env.sim, env.envs[0], camera_handle, seg_render_type
        )
    )
    segments = seg_image.flatten()

    return points, normals, segments


def farthest_point_sample_GPU(points, npoint):
    """
    Input:
        points: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """

    B, N, C = points.shape
    centroids = torch.zeros((B, npoint), dtype=torch.long, device=points.device)
    distance = torch.ones((B, N), device=points.device) * 1e10

    batch_indices = torch.arange(B, device=points.device)

    barycenter = torch.sum(points, dim=1) / N
    barycenter = barycenter.view(B, 1, C)

    dist = torch.sum((points - barycenter) ** 2, dim=-1)  # (B,N)
    farthest = torch.argmax(dist, dim=1)  # (B)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.argmax(distance, dim=1)
    # sampled_points = points[batch_indices, centroids, :]

    return centroids


def normal_to_direction(normal):

    z_axis = torch.tensor([0, 0, 1], device=normal.device, dtype=normal.dtype)
    cos_theta = torch.dot(normal, z_axis) / (torch.norm(normal) * torch.norm(z_axis))
    theta = torch.acos(cos_theta)

    angle_with_xy_plane = np.pi / 2 - theta

    normal_xy_projection = normal.clone()
    normal_xy_projection[2] = 0

    normal_xy_projection = normal.clone()
    normal_xy_projection[2] = 0

    x_axis = torch.tensor([1, 0, 0], device=normal.device, dtype=normal.dtype)
    y_axis = torch.tensor([0, 1, 0], device=normal.device, dtype=normal.dtype)

    # 使用 atan2 计算角度
    phi = torch.atan2(
        torch.dot(normal_xy_projection, y_axis), torch.dot(normal_xy_projection, x_axis)
    )

    return angle_with_xy_plane.item(), phi.item()
