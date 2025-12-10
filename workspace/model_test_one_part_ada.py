import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import furniture_bench
import gym
import torch
import numpy as np
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import *
import isaacgym
from isaacgym import gymapi, gymtorch
import argparse

import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import trange
import importlib

from utils.point_cloud import obs_to_pc_fps
from utils.direction import dir2pos
from tasks.utils import *
from tasks.task_config import all_task_config
from utils.distance import double_geodesic_distance_between_poses
from utils.train_utils import get_unique_dirname


from models.model_aff_assemble_ada import Network as AffNetwork
from models.model_actor_assemble_ada import Network as ActorNetwork
from models.model_critic_assemble_ada import Network as CriticNetwork

# without_interaction = False
save_ply = False

def load_model(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = {}
    network["device"] = device
    feat_dim = config["feat_dim"]
    cp_feat_dim = config["cp_feat_dim"]
    dir_feat_dim = config["dir_feat_dim"]

    network["aff"] = AffNetwork(feat_dim, cp_feat_dim).to(device)
    network["actor"] = ActorNetwork(feat_dim, cp_feat_dim, dir_feat_dim).to(device)
    network["critic"] = CriticNetwork(feat_dim, cp_feat_dim, dir_feat_dim).to(device)
    aff_ckpt = torch.load(config["aff_ckpt"], weights_only=True)
    actor_ckpt = torch.load(config["actor_ckpt"], weights_only=True)
    critic_ckpt = torch.load(config["critic_ckpt"], weights_only=True)
    # checkpoint = torch.load(config["model_path"])
    network["aff"].load_state_dict(aff_ckpt)
    network["actor"].load_state_dict(actor_ckpt)
    network["critic"].load_state_dict(critic_ckpt)

    network["aff"].to(device)
    network["actor"].to(device)
    network["critic"].to(device)

    return network


def random_sampled_points(pcs):
    point = torch.randint(0, pcs.shape[0], (1,))
    point = pcs[point].squeeze(0)
    result = sample_dir(point[:3], point[3:6], vertical=True)

    if result is False:
        return False
    else:
        hand_pos, hand_ori, action_dir = result
        return hand_pos, hand_ori, point, action_dir

def heuristic_sample(pcs,cur_pose,task,show=False):
    
    idx=heuristic_point_sample(pcs[:,:3],cur_pose,task ,show=show)
    point=pcs[idx].squeeze(0)
    result = sample_dir(point[:3], point[3:6],vertical=True)
    
    if result is False:
        return False
    else:
        hand_pos, hand_ori, action_dir = result
        return hand_pos, hand_ori, point, action_dir
  

def sample_hand_pose(network, pcs, cp1, interaction, points_num=2048, rvs=100, show=False):
    aff = network["aff"]
    actor = network["actor"]
    critic = network["critic"]
    device = network["device"]
    pcs = pcs.unsqueeze(0).to(device)
    cp1 = cp1.unsqueeze(0).to(device)
    a, b, c, d = interaction
    interaction = (
        a.unsqueeze(0).to(device),
        b.unsqueeze(0).to(device),
        c.unsqueeze(0).to(device),
        d.unsqueeze(0).to(device),
    )
    aff.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        scores = aff.inference_whole_pc(pcs, cp1, interaction).reshape(-1)[:points_num]
        if show:
            score_np = scores.cpu().numpy()
            score_normalized = (score_np - score_np.min()) / (
                score_np.max() - score_np.min()
            )

            all_points = pcs.cpu().numpy()[0, :, :3]

            part1_colors = plt.get_cmap("coolwarm")(score_normalized)[:, :3]
            part1_colors[:, 0] = score_normalized
            part2_colors = np.zeros((pcs.shape[1] - points_num, 3))
            all_colors = np.concatenate((part1_colors, part2_colors), axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(all_colors)

            o3d.visualization.draw_geometries([pcd])
            
        max_index = torch.argmax(scores).item()
        sampled_point = pcs[0, max_index].squeeze(0)
        
        if show:
            color=np.zeros((pcs.shape[1], 3))
            color[:,2]=1.0
            color[max_index]=[1.0, 0.0, 0.0]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(color)

            o3d.visualization.draw_geometries([pcd])
        
        
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
        action_direct = action_direct / torch.norm(action_direct)
        sampled_point = sampled_point.squeeze(0)

        hand_pos, hand_ori = dir2pos(sampled_point[:3], action_direct, True)

    return hand_pos, hand_ori, sampled_point, action_direct


def sample_hand_pose_multi(network, pcs, cp1, interaction, points_num=2048, rvs=100, show=False):
    aff = network["aff"]
    actor = network["actor"]
    critic = network["critic"]
    device = network["device"]
    pcs = pcs.unsqueeze(0).to(device)
    cp1 = cp1.unsqueeze(0).to(device)
    a, b, c, d = interaction
    interaction = (
        a.unsqueeze(0).to(device),
        b.unsqueeze(0).to(device),
        c.unsqueeze(0).to(device),
        d.unsqueeze(0).to(device),
    )
    aff.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        scores = aff.inference_whole_pc(pcs, cp1, interaction).reshape(-1)[:points_num]
        if show:
            score_np = scores.cpu().numpy()
            score_normalized = (score_np - score_np.min()) / (
                score_np.max() - score_np.min()
            )

            all_points = pcs.cpu().numpy()[0, :, :3]

            part1_colors = plt.get_cmap("coolwarm")(score_normalized)[:, :3]
            part1_colors[:, 0] = score_normalized
            part2_colors = np.zeros(((pcs.shape[1] - points_num) if points_num is not None else 0, 3))
            all_colors = np.concatenate((part1_colors, part2_colors), axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(all_colors)

            o3d.visualization.draw_geometries([pcd])
            if save_ply:
                save_ply_dir=get_unique_dirname("ply")
                if not os.path.exists(save_ply_dir):
                    os.makedirs(save_ply_dir)
                pcd.colors = o3d.utility.Vector3dVector()
                np.save(os.path.join(save_ply_dir, "score.npy"), score_np)
                np.save(os.path.join(save_ply_dir, "score_normalized.npy"), score_normalized)
                np.save(os.path.join(save_ply_dir,"pcs.npy"), pcs.cpu().numpy())
                o3d.io.write_point_cloud(
                    os.path.join(save_ply_dir, "points.ply"), pcd
                )
                print(f"Saved point cloud to {save_ply_dir}")
                # sys.exit(0)
                
                

        _, topk = torch.topk(scores, rvs)
        sampled_points = pcs[0, topk]
        recon_dir2 = actor.actor_sample_n_diffCtpts(
            pcs, cp1, sampled_points, interaction, rvs, rvs
        )
        action_score = critic.forward_n_diffCtpts(
            pcs, cp1, sampled_points, recon_dir2, interaction, rvs, rvs
        )
        ac_index = torch.argmax(action_score).item()
        action_direct = recon_dir2[ac_index]
        point_idx = ac_index // rvs
        sampled_point = sampled_points[point_idx]
        sampled_point = sampled_point.squeeze(0)
        hand_pos, hand_ori = dir2pos(sampled_point[:3], action_direct,True)

    return hand_pos, hand_ori, sampled_point, action_direct


def collect_data(task_config, network, collect=False, test_model="model",without_interaction=False):
    task_name = task_config["task_name"]
    REV = task_name.split("_")[-1] == "r" or task_name.split("_")[-1] == "pull"
    furniture_name = task_config["furniture_name"]
    part1_name = task_config["part_names"][0]

    save_test = False
    finetune_num = 32
    save_failed = False

    zip_save_finished = False
    env = gym.make(
        "dual-franka-hand-v2",
        furniture=furniture_name,
        num_envs=1,
        record=False,
        resize_img=False,
        assembled=REV,
        task_config=task_config,
    )
    env.set_obs(0)
    env.set_check_t(1)
    
    env.network = network

    terminated_count = 0
    finished_count = 0
    failed_count = 0
    success_count = 0

    base_data_path = f"data_test/{task_name}_{test_model}"+ (f"_woi" if without_interaction else "")
    finished_path = os.path.join(base_data_path, "finished/")
    failed_path = os.path.join(base_data_path, "failed/")

    if not os.path.exists(base_data_path):
        os.makedirs(base_data_path)
    if not os.path.exists(finished_path):
        os.makedirs(finished_path)
    else:
        finished_count = len(os.listdir(finished_path))
        # finished_count = 207

    if not os.path.exists(failed_path):
        os.makedirs(failed_path)
    else:
        failed_count = len(os.listdir(failed_path))

    finished_start = finished_count
    failed_count = failed_count
    finished_data = []
    
    # Start to perform the task
    module_name = task_name if task_name.split("_")[-1] != "r" else task_name[:-2]
    task_module = importlib.import_module(f"tasks.{module_name}")

    prepare = getattr(task_module, "prepare")
    perform = getattr(task_module, "perform")
    perform_r = getattr(task_module, "perform_r")
    
    reset_random_length = getattr(task_module, "reset_random_length")

    for collect_count in range(10000000):
        if save_test and failed_count + success_count >= finetune_num:
            break
        if collect_count % 50 == 0:
            print(
                f"Finished:{finished_count} | Failed:{failed_count} | Total:{finished_count+terminated_count}"
            )
        env.reset()
        env.need_reset = False
        # env.reset_check=False
        rb_states = env.rb_states  # rigid body states
        part_idxs = env.part_idxs

        env.refresh()

        control_hand(env, False)
        move_franka_away(env)
        
        reset_random_length()

        ori_part1_pose = C.to_homogeneous(
            rb_states[part_idxs[part1_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
        )

        

        cur_obs = env.observation[0][-1]

        result = obs_to_pc_fps(cur_obs, env.camera_matix[0], env.img_size, [1],npoint=4096 if not save_ply else 4096, show=save_ply)
        if result is False:
            print("Part out of view")
            terminated_count += 1
            continue
        else:
            points, normals = result

        part1_points = points[1]
        part1_normals = normals[1]



        pcs = torch.cat((part1_points, part1_normals), dim=1)

        target_ee_states = None
        target_ee_states, result = prepare(env, target_ee_states)
        # check if the part1 is moved

        start_part1_pose = C.to_homogeneous(
            rb_states[part_idxs[part1_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
        )

        if not result:
            print("Gripper Collision at Pre Grasp XY")
            continue
        if (
            double_geodesic_distance_between_poses(ori_part1_pose, start_part1_pose)
            > 1.0
        ):
            print("Part1 Moved")
            continue

        ee_pos, _, _ = target_ee_states[0]
        dist = torch.norm(part1_points - ee_pos, dim=1)
        min_index = torch.argmin(dist)
        cp1 = pcs[min_index]

        padding_c = torch.zeros((1, 6))
        padding_d = torch.zeros((1, 3))
        padding_p = torch.zeros((1, 2048, 6))
        padding_m = torch.zeros((1, 12))
        interaction_p = (padding_c, padding_d, padding_p, padding_m)

        if test_model == "model_single":
            hand_pos, hand_ori, sampled_point, action_direct = sample_hand_pose(
                network, pcs, cp1, interaction_p
            )
        elif test_model == "model":
            hand_pos, hand_ori, sampled_point, action_direct = sample_hand_pose_multi(
                network, pcs, cp1, interaction_p, points_num=None if save_ply else 4096, show=save_ply
            )
        elif test_model == "random":
            result = random_sampled_points(pcs)
            if result is False:
                print("Random Sample Failed")
                continue
            hand_pos, hand_ori, sampled_point, action_direct = result
        elif test_model == "heuristic":
            result = heuristic_sample(pcs, start_part1_pose, task_name)
            if result is False:
                print("Random Sample Failed")
                continue
            hand_pos, hand_ori, sampled_point, action_direct = result
        else:
            raise ValueError(f"Unknown model type: {test_model}")

        sampled_normal = sampled_point[3:6]
        sampled_point = sampled_point[:3]
        hand_pos -= 0.025 * action_direct.cpu().numpy()
        # Set hand pos
        try:
            env.set_hand_transform(hand_pos, hand_ori)
        except:
            print("Hand pos out of bound:", hand_pos, hand_ori)
            continue

        # hand_pos -= 0.065 * action_direct.cpu().numpy()
        # smooth_hand_transform(env, hand_pos, hand_ori)

        contact_flag = env.check_contact()

        if not contact_flag:
            control_hand(env, True)
            env.sampled_points.append(sampled_point)
            env.sampled_normals.append(sampled_normal)
            env.action_directs.append(action_direct)

            env.set_last_check_step(max(len(env.observation[0]) - 1, 0))
            env.set_check(True)
            if test_model == "random" or test_model == "heuristic":
                env.set_check(False)

            all_finished = True

            # Start screw
            all_finished = perform(env, target_ee_states)

            while env.need_reset:
                env.need_reset = False
                env.set_check(False)

                wait_close(env, 10)
                control_hand(env, False)
                env.smooth_hand_p = None
                env.smooth_hand_q = None
                env.set_hand_transform(np.array([0.0, 0.0, 1.5]), hand_ori)
                
                result = perform_r(env, target_ee_states)
                if not result:
                    print("Gripper Collision at Perform_r")
                    all_finished = False
                    break
                
                move_franka_away(env)
                control_hand(env, False)

                cur_step = len(env.observation[0]) - 1
                env.moved.append(cur_step)
                cur_obs = env.observation[0][-1]
                pre_obs = env.observation[0][env.last_check_step]

                result = obs_to_pc_fps(cur_obs, env.camera_matix[0], env.img_size, [1], npoint=4096 if not save_ply else 20480, show=save_ply)
                pre_result = obs_to_pc_fps(
                    pre_obs, env.camera_matix[0], env.img_size, [1]
                )
                if result is False or pre_result is False:
                    print("Part out of view")
                    terminated_count += 1
                    continue
                else:
                    points, normals = result
                    pre_points, pre_normals = pre_result

                part1_points = points[1]
                part1_normals = normals[1]

                pcs = torch.cat((part1_points, part1_normals), dim=1)

                cur_pose = cur_obs["part_pose"][part1_name]
                pre_pose = pre_obs["part_pose"][part1_name]
                cur_T = C.to_homogeneous(
                    cur_pose[:3],
                    C.quat2mat(cur_pose[3:7]),
                )
                pre_T = C.to_homogeneous(
                    pre_pose[:3],
                    C.quat2mat(pre_pose[3:7]),
                )
                relative_T = torch.linalg.inv(cur_T) @ pre_T

                i_cp2 = torch.cat((sampled_point, sampled_normal), dim=-1)
                i_dir2 = action_direct
                i_pcs = torch.cat((pre_points[1], pre_normals[1]), dim=-1)
                i_move = relative_T[:3].flatten()
                interaction = (
                    i_cp2.unsqueeze(0),
                    i_dir2.unsqueeze(0),
                    i_pcs.unsqueeze(0),
                    i_move.unsqueeze(0),
                )

                target_ee_states = None
                target_ee_states, result = prepare(env, target_ee_states)
                if not result:
                    all_finished = False
                    print("Gripper Collision at Prepare")
                    break

                ee_pos, _, _ = target_ee_states[0]
                dist = torch.norm(part1_points - ee_pos, dim=1)
                min_index = torch.argmin(dist)
                cp1 = pcs[min_index]

                if without_interaction:
                    interaction = interaction_p

                if test_model == "model_single":
                    hand_pos, hand_ori, sampled_point, action_direct = sample_hand_pose(
                        network, pcs, cp1, interaction
                    )
                elif test_model == "model":
                    hand_pos, hand_ori, sampled_point, action_direct = (
                        sample_hand_pose_multi(network, pcs, cp1, interaction, points_num=None if save_ply else 4096, show=save_ply)
                    )
                elif test_model == "random":
                    result = random_sampled_points(pcs)
                    if result is False:
                        print("Random Sample Failed")
                        continue
                    hand_pos, hand_ori, sampled_point, action_direct = result
                else:
                    raise ValueError(f"Unknown model type: {test_model}")

                hand_pos -= 0.025 * action_direct.cpu().numpy()
                try:
                    env.set_hand_transform(hand_pos, hand_ori)
                except:
                    print("Hand pos out of bound:", hand_pos, hand_ori)
                    all_finished = False
                    break
                # hand_pos -= 0.065 * action_direct.cpu().numpy()
                # smooth_hand_transform(env, hand_pos, hand_ori)
                contact_flag = env.check_contact()
                if contact_flag:
                    all_finished = False
                    print("Contact")
                    break
                sampled_normal = sampled_point[3:6]
                sampled_point = sampled_point[:3]

                control_hand(env, True)
                env.sampled_points.append(sampled_point)
                env.sampled_normals.append(sampled_normal)
                env.action_directs.append(action_direct)

                env.set_last_check_step(max(len(env.observation[0]) - 1, 0))
                env.set_check(True)
                all_finished = True
                all_finished = perform(env, target_ee_states)

            if all_finished:
                env.refresh()
                final_part1_pose = C.to_homogeneous(
                    rb_states[part_idxs[part1_name]][0][:3],
                    C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
                )
                final_z = final_part1_pose[2, 3]
                start_z = start_part1_pose[2, 3]
                
                distance = double_geodesic_distance_between_poses(
                    start_part1_pose, final_part1_pose
                )
                interaction = {
                    "cp2": [],
                    "dir2": [],
                    "pcs": [],
                    "move": [],
                    "pose1": [],
                    "pose2": [],
                }
                if len(env.moved) > 0:
                    for index, i in enumerate(env.moved):
                        if index % 3 == 0 or index % 3 == 1:
                            continue
                        index = index // 3
                        sampled_point = env.sampled_points[index]
                        sampled_normal = env.sampled_normals[index]
                        cp2 = torch.cat((sampled_point, sampled_normal), dim=0)
                        dir2 = env.action_directs[index]
                        interaction["cp2"].append(cp2.cpu())
                        interaction["dir2"].append(dir2.cpu())
                        cur_step = env.moved[index * 3 + 1]
                        pre_step = env.moved[index * 3]
                        pre_pose = env.observation[0][pre_step]["part_pose"][part1_name]
                        cur_pose = env.observation[0][cur_step]["part_pose"][part1_name]
                        cur_T = C.to_homogeneous(
                            cur_pose[:3],
                            C.quat2mat(cur_pose[3:7]),
                        )

                        pre_T = C.to_homogeneous(
                            pre_pose[:3],
                            C.quat2mat(pre_pose[3:7]),
                        )
                        interaction["pose1"].append(pre_T.cpu())
                        interaction["pose2"].append(cur_T.cpu())
                        relative_T = torch.linalg.inv(cur_T) @ pre_T
                        move = relative_T[:3].flatten()
                        interaction["move"].append(move.cpu())
                        points, normals = obs_to_pc_fps(
                            env.observation[0][pre_step],
                            env.camera_matix[0],
                            env.img_size,
                            [1],
                        )
                        pcs = torch.cat((points[1], normals[1]), dim=1)
                        interaction["pcs"].append(pcs.cpu())
                    distance = double_geodesic_distance_between_poses(
                        cur_T.cpu(), final_part1_pose.cpu()
                    )

                print("Distance: ", distance)
                height = final_part1_pose[2, 3] - start_part1_pose[2, 3]
                print("Height: ", height)

                write_data = np.array(
                    [
                        [
                            distance.item(),
                            height.item(),
                            final_part1_pose[0, 3].item(),
                            final_part1_pose[1, 3].item(),
                            final_part1_pose[2, 3].item(),
                        ]
                    ]
                )
                file_exist = os.path.exists(os.path.join(base_data_path, "log.csv"))
                with open(os.path.join(base_data_path, "log.csv"), "a") as f:
                    np.savetxt(
                        f,
                        write_data,
                        delimiter=",",
                        header=(
                            "Distance,Height,part1_pose_x,part1_pose_y,part1_pose_z"
                            if not file_exist
                            else ""
                        ),
                        fmt="%f",
                    )

                if env.record:
                    break
                cur_data = {
                    "all_finished": all_finished,
                    "part1_points": part1_points.cpu(),
                    "part1_normals": part1_normals.cpu(),
                    "sampled_point": env.sampled_points[-1].cpu(),
                    "sampled_normal": env.sampled_normals[-1].cpu(),
                    "action_direct": env.action_directs[-1].cpu(),
                    "start_part1_pose": start_part1_pose.cpu(),
                    "final_part1_pose": final_part1_pose.cpu(),
                    "distance": distance.cpu(),
                    "interaction": interaction,
                }
                # save data
                if save_test:
                    if zip_save_finished:
                        finished_data.append(cur_data)
                    else:
                        save_path = os.path.join(
                            finished_path, "data_%d.pt" % finished_count
                        )
                        torch.save(cur_data, save_path)
                finished_count += 1
            else:
                failed_count += 1
                terminated_count += 1
        else:
            write_data = np.array([[100.0, 0.0, 0, 0, 0]])
            file_exist = os.path.exists(os.path.join(base_data_path, "log.csv"))
            with open(os.path.join(base_data_path, "log.csv"), "a") as f:
                np.savetxt(
                    f,
                    write_data,
                    delimiter=",",
                    header=(
                        "Distance,Height,part1_pose_x,part1_pose_y,part1_pose_z"
                        if not file_exist
                        else ""
                    ),
                    fmt="%f",
                )
            if save_failed:
                interaction = {
                    "cp2": [],
                    "dir2": [],
                    "pcs": [],
                    "move": [],
                    "pose1": [],
                    "pose2": [],
                }
                distance = torch.tensor(100.0)
                print("Distance: ", distance)
                cur_data = {
                    "all_finished": True,
                    "part1_points": part1_points.cpu(),
                    "part1_normals": part1_normals.cpu(),
                    "sampled_point": sampled_point.cpu(),
                    "sampled_normal": sampled_normal.cpu(),
                    "action_direct": action_direct.cpu(),
                    "start_part1_pose": ori_part1_pose.cpu(),
                    "final_part1_pose": ori_part1_pose.cpu(),
                    "distance": distance,
                    "interaction": interaction,
                }
                if zip_save_finished:
                    finished_data.append(cur_data)
                else:
                    save_path = os.path.join(failed_path, "data_%d.pt" % failed_count)
                    torch.save(cur_data, save_path)
                failed_count += 1

        if finished_count >= 25:
            break

if __name__ == "__main__":
    # lamp|square_table|desk|drawer|cabinet|round_table|stool|chair|one_leg
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--task", type=str, default="cask", help="task name")
    argparse.add_argument(
        "--test_model",
        type=str,
        default="model",
        help="test model type: random|model_single|model|heuristic",
    )
    argparse.add_argument(
        "--without_interaction",
        action="store_true",
        help="whether to use interaction in model",
    )
    args = argparse.parse_args()
    task_name = args.task
    test_model = args.test_model
    without_interaction = args.without_interaction
    task_config = all_task_config[task_name]
    # random|model|Heuristic

    config = {
        "aff_ckpt": f"",
        "actor_ckpt": f"",
        "critic_ckpt": f"",
        "feat_dim": 128,
        "cp_feat_dim": 32,
        "dir_feat_dim": 32,
    }
    
    network=None

    if test_model == "model_single" or test_model == "model":
        network = load_model(config)

    collect_data(task_config, network, collect=False, test_model=test_model,without_interaction=without_interaction)
