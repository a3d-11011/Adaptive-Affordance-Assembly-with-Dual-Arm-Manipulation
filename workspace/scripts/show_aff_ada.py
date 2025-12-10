import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


from models.model_aff_assemble_ada import Network as AffordanceNetwork
from utils.direction import get_cp1
from utils.point_cloud import rotate_point_cloud,vis_pick_points

use_normals = True

if __name__ == "__main__":
    
    data_dir ="data/desk_cl/finished/data_131.pt"
    checkpoint_path = "/mnt/data/Dual-assemble/logs_cl/desk/aff_0/ckpts/58-network.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = AffordanceNetwork(
        128, 32, 9 if use_normals else 6, use_normals=use_normals
    )
    network.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    network.to(device)
    network.eval()

    data = torch.load(data_dir, weights_only=True)
    part1_points = data["part1_points"]
    part1_normals = data["part1_normals"]
    
    if "part2_points" in data.keys():
        part2_points = data["part2_points"]
        part2_normals = data["part2_normals"]
    
    n="6"
    points = np.load(f"ply_{n}/pcs.npy", allow_pickle=True)[0]
    points =torch.tensor(points, dtype=torch.float32)
    half=len(points)//2
    points[half:,:] = rotate_point_cloud(points[half:,:], -np.pi/6,"z")
    
    points[:,0]+=0.1
    points[:,1]+=0.1
    points[:,2]+=0.05
    
    half = points.shape[0] //2
    part1_points = torch.tensor(points[:half,:3], dtype=torch.float32)
    part1_normals = torch.tensor(points[:half,3:], dtype=torch.float32)
    part2_points = torch.tensor(points[half:,:3], dtype=torch.float32)
    part2_normals = torch.tensor(points[half:,3:], dtype=torch.float32)    
    print(len(data["interaction"]["move"]))

    if len(data["interaction"]["move"]) == 0:
        part1_pos = data["start_part1_pose"]
    else:
        part1_pos = data["interaction"]["pose1"][-1]

    sampled_point = data["sampled_point"]
    sampled_normal = data["sampled_normal"]
    action_direct = data["action_direct"]

    if "cp1" not in data.keys():
        idx=vis_pick_points(part2_points)[0]
        cp1=torch.cat((part2_points[idx], part2_normals[idx]), dim=-1)
    else:
        cp1 = data["cp1"]
    part1 = torch.cat((part1_points, part1_normals), dim=-1)
    if "part2_points" in data.keys():
        part2 = torch.cat((part2_points, part2_normals), dim=-1)
        pcd = torch.cat((part1, part2), dim=0).unsqueeze(0)
    else:
        pcd = part1.unsqueeze(0)
    cp1 = cp1.unsqueeze(0)

    pcd = pcd.to(device)
    cp1 = cp1.to(device)

    for i in range(2):
        if i == 0:
            print("No interaction data")
            i_move = torch.zeros((1, 12))
            i_cp2 = torch.zeros((1, 6))
            i_dir2 = torch.zeros((1, 3))
            i_pcs = torch.zeros((1, 2048, 6))
        else:
            if data["interaction"]["move"]:
                print("With interaction data")
                i_move = torch.stack(data["interaction"]["move"])
                i_cp2 = torch.stack(data["interaction"]["cp2"])
                i_dir2 = torch.stack(data["interaction"]["dir2"])
                i_pcs = torch.stack(data["interaction"]["pcs"])
                i_pcs = torch.nan_to_num(i_pcs, nan=0.0)
            else:
                break

        interact = (
            i_cp2.unsqueeze(0).to(device),
            i_dir2.unsqueeze(0).to(device),
            i_pcs.unsqueeze(0).to(device),
            i_move.unsqueeze(0).to(device),
        )

        with torch.no_grad():
            if not network.use_normals:
                pcd = pcd[:, :, :3]
                cp1 = cp1[:, :3]
                interact = (
                    interact[0][:, :, :3],
                    interact[1],
                    interact[2][:, :, :, :3],
                    interact[3],
                )
            pcd = torch.nan_to_num(pcd, nan=0.0)
            pred_score = network.inference_whole_pc(pcd, cp1, interact)
        if i==0:
            np.save(f"ply_{n}/score_.npy",pred_score[0].cpu().numpy())
        if i==1:
            np.save(f"ply_{n}/score_a.npy", pred_score[0].cpu().numpy())
        
        pred_score = pred_score.reshape(-1)[: part1_points.shape[0]]
        score_np = pred_score.cpu().numpy()
        score_normalized = (score_np - score_np.min()) / (
            score_np.max() - score_np.min()
        )

        part1_np = part1_points.cpu().numpy()
        if "part2_points" in data.keys():
            part2_np = part2_points.cpu().numpy()
            all_points = np.concatenate((part1_np, part2_np), axis=0)
        else:
            all_points = part1_np

        part1_colors = plt.get_cmap("coolwarm")(score_normalized)[:, :3]
        part1_colors[:, 0] = score_normalized
        if "part2_points" in data.keys():
            part2_colors = np.zeros((part2_np.shape[0], 3))
            all_colors = np.concatenate((part1_colors, part2_colors), axis=0)
        else:
            all_colors = part1_colors

        pcd_s = o3d.geometry.PointCloud()
        pcd_s.points = o3d.utility.Vector3dVector(all_points)
        pcd_s.colors = o3d.utility.Vector3dVector(all_colors)

        o3d.visualization.draw_geometries([pcd_s])
