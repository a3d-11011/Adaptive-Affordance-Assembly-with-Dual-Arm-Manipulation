import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from rotation import rot_mat, rot_mat_to_angles_tensor


def get_cp1_dir1(part2_points: torch.Tensor, part2_pose=None, task="desk"):
    device = part2_points.device
    if (
        task.startswith("desk")
        or task.startswith("lamp")
        or task.startswith("square_table")
        or task.startswith("chair")
        or task.startswith("round_table")
        or task.startswith("stool")
    ):
        centorids = part2_points[:, :3].mean(dim=0)
        tolerance = 0.01
        mask = (
            (torch.abs(part2_points[:, 0] - centorids[0]) < tolerance)
            & (torch.abs(part2_points[:, 1] - centorids[1]) < tolerance)
            & (part2_points[:, 2] > centorids[2])
        )
        cp1 = part2_points[mask]
        if len(cp1) == 0:
            torch.save(part2_points, "erroe_part2_points.pt")
            raise ("no cp1 found!")
        cp1 = cp1[0]
        cp1_normal = torch.zeros([3], device=device)
        cp1_normal[2] = 1.0
        cp1 = torch.cat((cp1, cp1_normal), dim=-1)
        dir1 = torch.tensor([0.0, 0.0, 0.0], device=device)
    elif task.startswith("drawer") or task.startswith("cabinet"):
        part2_angle = rot_mat_to_angles_tensor(part2_pose[:3, :3], device)
        part2_forward = part2_angle[2].item()
        part2_pos = part2_pose[:3, 3]
        part2_forward = torch.tensor(
            [-np.sin(part2_forward), np.cos(part2_forward), 0.0],
            dtype=torch.float32,
            device=device,
        )
        diff = part2_points[:, :3] - part2_pos
        proj = torch.matmul(diff, part2_forward)
        idx = torch.argmax(proj)
        # _,idx=torch.topk(proj,10)
        # idx=idx[torch.randint(0,10,(1,))][0]
        cp1 = part2_points[idx]
        dir1 = torch.tensor([0.0, 0.0, 0.0], device=device)
    else:
        raise ("not define to get cp1 and dir1!")

    return cp1, dir1


def get_cp1(part2_points, part1_pose=None, task="desk"):
    device = part2_points.device
    if (
        task.startswith("desk")
        or task.startswith("lamp")
        or task.startswith("square_table")
        or task.startswith("chair")
        or task.startswith("round_table")
        or task.startswith("stool")
    ):
        part2_pose = part2_points[:, :3].mean(dim=0)
        part2_forward = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=device
        )
        diff = part2_points[:, :3] - part2_pose
        proj = torch.matmul(diff, part2_forward)
        idx = torch.argmax(proj)
        cp1 = part2_points[idx]
        # part2_pose=part2_points[:,:3].mean(dim=0)
        # tolerance = 0.03
        # mask = (torch.abs(part2_points[:, 0] - part2_pose[0]) < tolerance) & (torch.abs(part2_points[:, 1] - part2_pose[1]) < tolerance) & (part2_points[:, 2] > part2_pose[2])
        # cp1=part2_points[mask]
        # dis=torch.norm(cp1[:,:2]-part2_pose[:2],dim=1)
        # min_dix=torch.argmin(dis)
        # if len(cp1)==0:
        #     torch.save(part2_points,"erroe_part2_points.pt")
        #     raise("no cp1 found!")

        # cp1=cp1[min_dix]
    elif task.startswith("drawer") or task.startswith("cabinet"):
        part2_angle = rot_mat_to_angles_tensor(part1_pose[:3, :3], device)
        part2_forward = part2_angle[2].item()
        part2_pos = part1_pose[:3, 3]
        part2_forward = torch.tensor(
            [-np.sin(part2_forward), np.cos(part2_forward), 0.0],
            dtype=torch.float32,
            device=device,
        )
        diff = part2_points[:, :3] - part2_pos
        proj = torch.matmul(diff, part2_forward)
        idx = torch.argmax(proj)
        # _,idx=torch.topk(proj,5)
        # idx=idx[torch.randint(0,5,(1,))][0]
        cp1 = part2_points[idx]
    else:
        raise ("not define to get cp1!")

    return cp1


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


def dir2pos_fine(sampled_point: torch.Tensor, act_dir: torch.Tensor):
    sampled_p = sampled_point[0:3]
    sampled_n = sampled_point[3:6]
    s_a, phi = normal_to_direction(sampled_n)
    sampled_angle_with_xy_plane, phi_ = normal_to_direction(act_dir)

    hand_ori = rot_mat([0, np.pi / 2 + sampled_angle_with_xy_plane, 0], hom=True)
    hand_ori = rot_mat([0, 0, phi - np.pi], hom=True) @ hand_ori

    normal_xy_projection = torch.sqrt(sampled_n[0] ** 2 + sampled_n[1] ** 2)
    sampled_angle_with_xy_plane = torch.tensor(
        sampled_angle_with_xy_plane, device=sampled_n.device
    )
    action_z = normal_xy_projection * torch.abs(
        torch.sin(sampled_angle_with_xy_plane) / torch.cos(sampled_angle_with_xy_plane)
    )

    action_direct = sampled_n.clone()
    if sampled_angle_with_xy_plane > np.pi / 2:
        action_direct = -action_direct
    action_direct[2] = action_z
    action_direct = action_direct / torch.norm(action_direct)

    dev = 0.0045 * np.sin(sampled_angle_with_xy_plane.cpu() - s_a)

    hand_pos = sampled_point[0:3] + action_direct * (0.12 + dev)
    hand_pos = hand_pos.cpu().numpy()

    return hand_pos, hand_ori, action_direct


def dir2pos(sampled_point: torch.Tensor, act_dir: torch.Tensor, vertical=False):

    sampled_angle_with_xy_plane, phi = normal_to_direction(act_dir)

    if vertical:
        hand_ori = rot_mat([np.pi / 2 + sampled_angle_with_xy_plane, 0, 0], hom=True)
        hand_ori = rot_mat([0, 0, phi - np.pi / 2], hom=True) @ hand_ori
    else:
        hand_ori = rot_mat([0, np.pi / 2 + sampled_angle_with_xy_plane, 0], hom=True)
        hand_ori = rot_mat([0, 0, phi - np.pi], hom=True) @ hand_ori
    dev = 0.0045 * np.random.uniform(-0.5, 0.5)
    hand_pos = sampled_point + act_dir * (0.12 + dev)
    hand_pos = hand_pos.cpu().numpy()

    return hand_pos, hand_ori


def sample_dir(
    sampled_point: torch.Tensor,
    sampled_normal: torch.Tensor,
    attempt=1000,
    vertical=False,
):
    if torch.isnan(sampled_normal).any() or torch.count_nonzero(sampled_normal) == 0:
        return False
    angle_with_xy_plane, phi = normal_to_direction(sampled_normal)
    sampled_angle_with_xy_plane = -1
    num_trials = 0
    if not vertical:
        while sampled_angle_with_xy_plane < 0:
            # if angle_with_xy_plane <np.pi/6:
            #     random_angle = np.random.uniform(0, np.pi/6)
            # else:
            random_angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            sampled_angle_with_xy_plane = angle_with_xy_plane + random_angle
            num_trials += 1
            if num_trials > attempt:
                return False
    else:
        if angle_with_xy_plane < 0:
            angle_with_xy_plane = 0
        random_angle = np.random.uniform(-np.pi / 9, np.pi / 9)
        sampled_angle_with_xy_plane = angle_with_xy_plane + random_angle
        sampled_angle_with_xy_plane = np.random.uniform(
            np.pi / 2 - np.pi / 180, np.pi / 2 - np.pi / 360
        )
    print("sampled_normal:", sampled_normal)
    print("angle_with_xy_plane:", angle_with_xy_plane / np.pi * 180)
    print("random_angle:", random_angle / np.pi * 180)
    print("sampled_angle_with_xy_plane:", sampled_angle_with_xy_plane / np.pi * 180)
    if vertical:
        # hand_ori = rot_mat([-np.pi/2-sampled_angle_with_xy_plane, 0, 0], hom=True)
        # hand_ori = rot_mat([0, 0, phi+np.pi/2], hom=True) @ hand_ori
        hand_ori = rot_mat([np.pi / 2 + sampled_angle_with_xy_plane, 0, 0], hom=True)
        hand_ori = rot_mat([0, 0, phi - np.pi / 2], hom=True) @ hand_ori

    else:
        hand_ori = rot_mat([0, np.pi / 2 + sampled_angle_with_xy_plane, 0], hom=True)
        hand_ori = rot_mat([0, 0, phi - np.pi], hom=True) @ hand_ori

    print(
        "hand_ori:",
        [
            (np.pi / 2 + sampled_angle_with_xy_plane) / np.pi * 180,
            0,
            (phi - np.pi / 2) / np.pi * 180,
        ],
    )
    print(
        "hand_ori:",
        [
            0,
            (np.pi / 2 + sampled_angle_with_xy_plane) / np.pi * 180,
            (phi - np.pi) / np.pi * 180,
        ],
    )
    normal_xy_projection = torch.sqrt(sampled_normal[0] ** 2 + sampled_normal[1] ** 2)
    sampled_angle_with_xy_plane = torch.tensor(
        sampled_angle_with_xy_plane, device=sampled_normal.device
    )
    action_z = normal_xy_projection * torch.tan(
        sampled_angle_with_xy_plane
    )  # torch.abs(torch.sin(sampled_angle_with_xy_plane)/torch.cos(sampled_angle_with_xy_plane))

    action_direct = sampled_normal.clone()
    # if sampled_angle_with_xy_plane > np.pi/2:
    #     action_direct = -action_direct
    action_direct[2] = action_z
    action_direct = action_direct / torch.norm(action_direct)

    dev = 0.0045 * np.sin(random_angle)

    dis = 0.125
    # if vertical:
    #     dis=0.12
    hand_pos = sampled_point + action_direct * (dis + dev)
    hand_pos = hand_pos.cpu().numpy()
    # if vertical:
    #     hand_pos[2] = hand_pos[2]+0.02
    # hand_pos,hand_ori=dir2pos(sampled_point,action_direct)
    print("action_direct:", action_direct.cpu().numpy())
    return hand_pos, hand_ori, action_direct


def sample_dir2(sampled_point: torch.Tensor, sampled_normal: torch.Tensor):
    angle_with_xy_plane, phi = normal_to_direction(sampled_normal)
    sampled_angle_with_xy_plane = -1
    num_trials = 0

    sample_angles = [
        np.pi / 6,
        np.pi / 6 + np.pi / 18,
        np.pi / 6 + np.pi / 9,
        np.pi / 3,
    ]

    idx = np.random.randint(0, len(sample_angles))

    sampled_angle_with_xy_plane = sample_angles[idx]

    hand_ori = rot_mat([0, np.pi / 2 + sampled_angle_with_xy_plane, 0], hom=True)
    hand_ori = rot_mat([0, 0, phi - np.pi], hom=True) @ hand_ori

    normal_xy_projection = torch.sqrt(sampled_normal[0] ** 2 + sampled_normal[1] ** 2)
    sampled_angle_with_xy_plane = torch.tensor(
        sampled_angle_with_xy_plane, device=sampled_normal.device
    )
    action_z = normal_xy_projection * torch.abs(
        torch.sin(sampled_angle_with_xy_plane) / torch.cos(sampled_angle_with_xy_plane)
    )

    action_direct = sampled_normal.clone()
    if sampled_angle_with_xy_plane > np.pi / 2:
        action_direct = -action_direct
    action_direct[2] = action_z
    action_direct = action_direct / torch.norm(action_direct)

    dev = 0.0045 * np.random.uniform(-0.5, 0.5)

    hand_pos = sampled_point + action_direct * (0.125 + dev)
    hand_pos = hand_pos.cpu().numpy()

    return hand_pos, hand_ori, action_direct
