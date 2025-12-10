import os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation, Slerp
import torch
from rotation import rotation_distance


def dense_sample_num(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_quat: np.ndarray,
    end_quat: np.ndarray,
    slow=True,
):
    """
    generate dense sample number for inverse kinematics control.
    """

    distance_p = np.linalg.norm(end_pos - start_pos).item()
    distance_q = rotation_distance(start_quat, end_quat)
    if slow:

        dense_sample_num = int(max(distance_p // 0.0025, distance_q // (np.pi / 90)))
    else:
        dense_sample_num = int(max(distance_p // 0.025, distance_q // (np.pi / 36)))

    return dense_sample_num


def dense_trajectory_points_generation(
    start_pos, end_pos, start_quat, end_quat, slow=True, num_points: int = None
):
    """
    generate dense trajectory points for inverse kinematics control.
    """
    if num_points is None:
        num_points = dense_sample_num(start_pos, end_pos, start_quat, end_quat, slow)

    if num_points <= 0:
        return [end_pos], [end_quat]
    # ---- 1. 生成五个采样点（包括起点和终点） ----
    # distance = np.linalg.norm(end_pos - start_pos)
    # distance = torch.norm(torch.tensor(end_pos).to("cuda:0") - torch.tensor(start_pos)).item()
    # print(distance)
    if np.linalg.norm(end_pos - start_pos) > 0.001:
        initial_sample_points_num = 5
        initial_sample_points = np.linspace(
            start_pos, end_pos, initial_sample_points_num
        )

        # ---- 2. 使用 B 样条拟合采样点，生成平滑轨迹 ----
        tck, u = splprep(initial_sample_points.T, s=0)  # B样条拟合
        u_new = np.linspace(0, 1, num_points)  # 更细致的采样
        interp_pos = np.array(splev(u_new, tck)).T  # 插值后的平滑轨迹
    else:
        interp_pos = np.array([start_pos] * num_points).squeeze()

    # ---- 3. 对旋转四元数进行球面线性插值 (Slerp) ----

    rotations = Rotation.from_quat(
        [start_quat.tolist(), end_quat.tolist()]
    )  # 四元数转换为旋转对象
    slerp = Slerp([0, 1], rotations)  # Slerp 插值器
    interp_times = np.linspace(0, 1, num_points)  # 插值时间点
    interp_rotations = slerp(interp_times).as_quat()  # 插值结果转换为四元数

    # interp_pos = torch.tensor(interp_pos)
    # interp_rotations = torch.tensor(interp_rotations)
    interp_pos = np.append(interp_pos, [end_pos], axis=0)
    interp_rotations = np.append(interp_rotations, [end_quat], axis=0)
    return interp_pos, interp_rotations
