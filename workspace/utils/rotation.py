import numpy as np
import torch


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


def rot_mat_tensor(x, y, z, device):
    return torch.tensor(rot_mat([x, y, z], hom=True), device=device).float()


def rel_rot_mat(s, t):
    s_inv = torch.linalg.inv(s)
    return t @ s_inv


def rotation_distance(q0, q1):
    """
    Compute the rotation distance between two quaternions.
    Args:
        q0: First quaternion.
        q1: Second quaternion.
    Returns:
        distance: Rotation distance.
    """
    q0 = torch.tensor(q0).float().cpu()
    q1 = torch.tensor(q1).float().cpu()
    q0 = q0 / torch.norm(q0)
    q1 = q1 / torch.norm(q1)
    dot = torch.dot(q0, q1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = 2 * torch.acos(dot)
    return angle


def euler_from_deltaR(R):
    # roll = da, pitch = db, yaw = dc
    da = torch.arctan2(R[2, 1], R[2, 2])
    db = torch.arcsin(-R[2, 0])
    dc = torch.arctan2(R[1, 0], R[0, 0])
    return da, db, dc


def is_tipped(R: torch.Tensor, theta_thresh_deg: float = 15.0,z=2,down=True ):
    u_world = R[:, z]
    if down:
        u_world = -u_world
    u_world = u_world / u_world.norm()
    g = torch.tensor([0.0, 0.0, 1.0], dtype=R.dtype, device=R.device)
    cos_theta = torch.clamp(torch.dot(u_world, g), -1.0, 1.0)
    theta = torch.acos(cos_theta)
    theta_deg = torch.rad2deg(theta)
    return theta_deg > theta_thresh_deg, theta_deg
