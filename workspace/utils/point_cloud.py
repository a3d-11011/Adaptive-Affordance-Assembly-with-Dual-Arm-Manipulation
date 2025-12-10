import torch
import numpy as np
import open3d as o3d


def rotate_pc_relative_to_origin(points, T):
    """
    rotate point cloud relative to origin point
    """
    rot_mat = T[:3, :3]
    pos = T[:3, 3]

    origin_points = points - pos
    rotate_points = origin_points @ rot_mat

    return rotate_points


def rotate_pc_to_origin_bach(points, T):
    """
    rotate point cloud to origin point
    """
    if len(points.shape) != 3:
        raise ValueError("points should be batch*n*3")
    batch_size = points.shape[0]
    T_len = len(T.shape)
    if T_len == 2:
        # apply T to all batch_points
        rot_mat = T[:3, :3]
        rot_xyz = T[:3, 3]
        ori_points = points.reshape(-1, 3)
        ori_points = ori_points - rot_xyz
        ori_points = ori_points @ rot_mat
        ori_points = ori_points.reshape(batch_size, -1, 3)
    elif T_len == 3:
        # apply T to each batch_points
        rot_mat = T[:, :3, :3]
        rot_xyz = T[:, :3, 3].unsqueeze(1)
        ori_points = points - rot_xyz
        ori_points = torch.matmul(ori_points, rot_mat)
    else:
        raise ValueError("T should be 2 or 3 dimensions")
    return ori_points


def rotate_points_to_pos(points, T):
    """
    rotate point cloud to pos
    """
    rot_mat = T[:3, :3]
    rot_xyz = T[:3, 3]
    rot_points = points @ rot_mat.T + rot_xyz
    return rot_points


def normalize_pc(points: torch.Tensor):
    """
    normalize point cloud
    """
    nsample = points.shape[0]
    centroid = torch.mean(points, dim=0)
    points = points - centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(points**2, dim=-1)))
    points = points / furthest_distance
    return points, centroid, furthest_distance


def normalize_pc_batch(points: torch.Tensor):
    """
    normalize point cloud batch
    """
    batch_size = points.shape[0]
    nsample = points.shape[1]
    centroid = torch.mean(points, dim=1, keepdim=True)
    points = points - centroid
    furthest_distance = torch.max(
        torch.sqrt(torch.sum(points**2, dim=-1, keepdim=True)), dim=1, keepdim=True
    )[0]
    points = points / furthest_distance
    return points, centroid, furthest_distance


def obs_to_pc_fps(
    obs,
    cam_mat,
    img_size,
    seg,
    npoint=2048,
    device: torch.device = "cuda:0",
    show=False,
):
    points = {}
    normals = {}
    _points, _normals, _segments = obs_to_pc(obs, cam_mat, img_size, device)
    for i in seg:
        pi = _points[_segments == i].unsqueeze(0)
        ni = _normals[_segments == i]
        if pi.shape[1] <= npoint:
            return False
        if npoint >0:
            fps_idxs = farthest_point_sample_GPU(pi, npoint).squeeze(0)
        else:
            fps_idxs = torch.arange(pi.shape[1], device=device)
        points[i] = pi[0][fps_idxs, :]
        normals[i] = ni[fps_idxs, :]

    if show:
        pcs = None
        for i in seg:
            if pcs is None:
                pcs = points[i][:, :3]
            else:
                pcs = torch.cat((pcs, points[i][:, :3]), dim=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs.cpu().numpy())
        o3d.visualization.draw_geometries([pcd])
    return points, normals


def obs_to_pc(obs, cam_mat, img_size, device: torch.device = "cuda:0"):
    points = None
    normals = None
    segments = None
    width = img_size[0]
    height = img_size[1]
    for cam_name, image in obs["depth_img"].items():
        if cam_name=='top':
            continue
        cam_proj = torch.tensor(cam_mat[cam_name]["proj"]).to(device)
        cam_view = torch.tensor(cam_mat[cam_name]["view"]).to(device)

        _points, _normals = depth_image_to_point_cloud_GPU(
            image.to(device),
            cam_view,
            cam_proj,
            width=width,
            height=height,
        )
        if points is None:
            points = _points
            normals = _normals
            segments = obs["segments"][cam_name].to(device)
        else:
            points = torch.cat((points, _points), dim=0)
            normals = torch.cat((normals, _normals), dim=0)
            segments = torch.cat(
                (segments, obs["segments"][cam_name].to(device)), dim=0
            )

    normals[torch.isnan(normals).any(dim=-1)] = torch.tensor(
        [0.0, 0.0, 0.0], device=normals.device
    )

    return points, normals, segments


def obs_to_pc_signal(depth_img, cam_mat, img_size, device: torch.device = "cuda:0"):
    width = img_size[0]
    height = img_size[1]
    cam_proj = torch.tensor(cam_mat["proj"]).to(device)
    cam_view = torch.tensor(cam_mat["view"]).to(device)

    points, normals = depth_image_to_point_cloud_GPU(
        depth_img.to(device),
        cam_view,
        cam_proj,
        width=width,
        height=height,
    )

    return points, normals


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


def compute_point_pos(img_pos, img, img_size, cam_mat, device: torch.device = "cuda:0"):
    img_width = img_size[0]
    img_height = img_size[1]
    index = img_pos[0] * img_width + img_pos[1]
    points, normals = obs_to_pc_signal(img, cam_mat, img_size, device=device)
    point = points[index]
    normal = normals[index]
    return point, normal


def find_farthest_nearby_point_torch(
    points: torch.Tensor, p0: torch.Tensor, v: torch.Tensor, d_max: float
):
    v_hat = v / v.norm()
    w = points[:, :3] - p0  # (N,3)
    t = (w * v_hat).sum(dim=1)  # (N,)

    proj = t.unsqueeze(1) * v_hat  # (N,3)
    d = (w - proj).norm(dim=1)  # (N,)

    mask = (t >= 0) & (d <= d_max)
    if not mask.any():
        return None, None

    candidates = points[mask][:, :3]  # (M,3)
    dist0 = (candidates - p0).norm(dim=1)  # (M,)
    far_in_cand = torch.argmax(dist0)  # 候选集内索引c

    farthest_idx = torch.nonzero(mask, as_tuple=False)[far_in_cand].item()
    farthest_pt = points[farthest_idx]

    return farthest_idx, farthest_pt


def find_max_perp_at_proj(
    points: torch.Tensor,
    p0: torch.Tensor,
    v: torch.Tensor,
    target_dist: float,
    tol: float = 0.01,
):
    v_hat = v / v.norm()
    w = points[:, :3] - p0  # (N,3)
    t = (w * v_hat).sum(dim=1)  # (N,)

    proj = t.unsqueeze(1) * v_hat  # (N,3)
    d = (w - proj).norm(dim=1)  # (N,)
    mask1 = (t >= target_dist - tol) & (t <= target_dist + tol)
    mask2 = (t >= -target_dist - tol) & (t <= -target_dist + tol)

    if not mask1.any() or not mask2.any():
        return None

    perps1 = d[mask1]  # (M,)
    idx_in_mask1_max = torch.argmax(perps1)
    idx_in_mask1_min = torch.argmin(perps1)

    perps2 = d[mask2]  # (M,)
    idx_in_mask2_max = torch.argmax(perps2)
    idx_in_mask2_min = torch.argmin(perps2)

    repro1 = torch.nonzero(mask1, as_tuple=False)
    repro2 = torch.nonzero(mask2, as_tuple=False)

    max_idx1 = repro1[idx_in_mask1_max].item()
    # min_idx1 = repro1[idx_in_mask1_min].item()
    max_idx2 = repro2[idx_in_mask2_max].item()
    # min_idx2 = repro2[idx_in_mask2_min].item()

    idx = torch.tensor([max_idx1, max_idx2], device=points.device)

    pt = points[idx]

    return idx, pt


def find_four_by_proj_and_perp(
    points: torch.Tensor,
    p0: torch.Tensor,
    v: torch.Tensor,
    target_dist: float,
    tol: float = 0.005,
    show=False,
):
    pts_xy = points[:, :2]  # (N,2)
    p0_xy = p0[:2]  # (2,)
    v_xy = v[:2]  # (2,)

    norm_v = v_xy.norm()
    if norm_v == 0:
        raise ValueError("输入方向向量 v 在 XY 平面长度为 0")

    v_hat = v_xy / norm_v
    w_xy = pts_xy - p0_xy
    t = (w_xy * v_hat).sum(dim=1)  # (N,)

    mask_pos = (t >= target_dist - tol) & (t <= target_dist + tol)
    mask_neg = (t >= -target_dist - tol) & (t <= -target_dist + tol)

    idxs_pos = torch.nonzero(mask_pos, as_tuple=False).squeeze(1)
    idxs_neg = torch.nonzero(mask_neg, as_tuple=False).squeeze(1)
    if show:
        pcd= o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].cpu().numpy())
        color = np.zeros((points.shape[0], 3))
        color[mask_pos.cpu().numpy(), 0]= 1.0
        color[mask_neg.cpu().numpy(), 2]= 1.0
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])
    if idxs_pos.numel() == 0 or idxs_neg.numel() == 0:
        return None, None
    # return idxs_pos,idxs_neg
    v2 = torch.tensor([-v_hat[1], v_hat[0]], device=points.device)

    s = (w_xy * v2).sum(dim=1)

    s_pos = s[idxs_pos]
    max_pos_idx = idxs_pos[torch.argmax(s_pos)].item()
    min_pos_idx = idxs_pos[torch.argmin(s_pos)].item()
    s_neg = s[idxs_neg]
    max_neg_idx = idxs_neg[torch.argmax(s_neg)].item()
    min_neg_idx = idxs_neg[torch.argmin(s_neg)].item()

    idx = torch.tensor(
        [max_pos_idx, min_pos_idx, max_neg_idx, min_neg_idx], device=points.device
    )  # (4,)
    pts = points[idx]  # (4,3)

    return idx, pts

def rotate_point_cloud(
    points: torch.Tensor,
    angle_deg: float,
    axis: str = 'z'
) -> torch.Tensor:
    """
    Rotate a point cloud tensor that may contain positions only (N×3)
    or positions + normals (N×6). The same rotation is applied to both.

    Args:
        points (torch.Tensor): shape (N,3) or (N,6), dtype float, any device.
        angle_deg (float): rotation angle in degrees.
        axis (str): one of 'x', 'y', or 'z'. Default is 'z'.

    Returns:
        torch.Tensor: rotated points, same shape as input.
    """
    # 输入校验
    assert points.ndim == 2 and points.shape[1] in (3, 6), \
        "points 应该是形状 (N,3) 或 (N,6)"
    dtype = points.dtype
    device = points.device

    # 角度转弧度
    theta = torch.deg2rad(torch.tensor(angle_deg, dtype=dtype, device=device))

    # 构造旋转矩阵 R (3×3)
    if axis.lower() == 'x':
        R = torch.tensor([
            [1,              0,               0],
            [0, torch.cos(theta), -torch.sin(theta)],
            [0, torch.sin(theta),  torch.cos(theta)]
        ], dtype=dtype, device=device)
    elif axis.lower() == 'y':
        R = torch.tensor([
            [ torch.cos(theta), 0, torch.sin(theta)],
            [               0, 1,               0],
            [-torch.sin(theta), 0, torch.cos(theta)]
        ], dtype=dtype, device=device)
    else:  # 'z'
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta),  torch.cos(theta), 0],
            [              0,                0, 1]
        ], dtype=dtype, device=device)

    # 分离坐标 (N×3) 和法线 (N×3)
    coords = points[:, :3]        # 位置
    coords_rot = coords @ R.T     # 旋转坐标

    # 如果有法线，做相同的旋转
    if points.shape[1] == 6:
        normals = points[:, 3:6]
        normals_rot = normals @ R.T
        return torch.cat([coords_rot, normals_rot], dim=1)

    return coords_rot

def vis_pick_points(points):
    import open3d as o3d
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(points.cpu().numpy())
    color=np.zeros((points.shape[0],3))
    pcd.colors=o3d.utility.Vector3dVector(color)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Pick points: Shift+LMB to pick, Q/Esc to finish",
        width=800, height=600
    )
    vis.add_geometry(pcd)

    # 4) Run the interaction loop (blocks until you close the window with Q or Esc):
    vis.run()

    # 5) Fetch the indices of the points you Shift+clicked:
    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    print("Picked indices:", picked_indices)
    return picked_indices