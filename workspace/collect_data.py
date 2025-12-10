import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import furniture_bench
import gym
import torch
import numpy as np
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import *
import isaacgym
from isaacgym import gymapi, gymtorch

import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import trange
import importlib
import argparse

from utils.point_cloud import obs_to_pc_fps
from utils.direction import sample_dir
from tasks.utils import *
from tasks.task_config import all_task_config
from utils.distance import double_geodesic_distance_between_poses


def collect_data(task_config):
    task_name = task_config["task_name"]
    REV = task_name.split("_")[-1] == "r" or task_name.split("_")[-1] == "pull"
    furniture_name = task_config["furniture_name"]
    part1_name = task_config["part_names"][0]
    part2_name = task_config["part_names"][1]

    save = False
    save_failed = False
    env = gym.make(
        "dual-franka-hand-v2",
        furniture=furniture_name,
        num_envs=1,
        # record=True,
        resize_img=False,
        assembled=REV,
        randomness="none",
        task_config=task_config,
        headless=False,
    )
    env.set_obs(0)
    env.set_check_t(1)

    terminated_count = 0
    finished_count = 0
    failed_count = 0

    zip_save_finished = False

    save_num = 1000

    base_data_path = f"data/{task_name}_cl"
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
    terminated_start = terminated_count
    finished_data = []
    terminated_data = []

    for collect_count in range(10000000):
        if failed_count >= 10000:
            save_failed = False
        if collect_count % 50 == 0:
            print(
                f"Finished:{finished_count} | Terminated:{terminated_count} | Total:{finished_count+terminated_count}"
            )
        env.reset()
        env.reset_failed = False
        env.need_reset = False
        # env.reset_check=False
        rb_states = env.rb_states  # rigid body states
        part_idxs = env.part_idxs

        env.refresh()
        small_wait(env)

        ori_part1_pose = C.to_homogeneous(
            rb_states[part_idxs[part1_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
        )
        ori_part2_pose = C.to_homogeneous(
            rb_states[part_idxs[part2_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
        )

        env.isaac_gym.start_access_image_tensors(env.sim)
        camera_names = ["front", "back", "left", "right"]
        points = None
        normals = None
        part2_points = None
        part2_normals = None

        for camera_name in camera_names:
            camera_handle = env.camera_handles[camera_name][0]
            points_, normals_, segments = capture_pc_headless(env, camera_handle)
            part2_points_ = points_[segments == 2]
            part2_normals_ = normals_[segments == 2]

            mask = segments == 1
            points_ = points_[mask]
            normals_ = normals_[mask]

            if points is None:
                points = points_
                normals = normals_
                part2_points = part2_points_
                part2_normals = part2_normals_
            else:
                points = torch.cat((points, points_), dim=0)
                normals = torch.cat((normals, normals_), dim=0)
                part2_points = torch.cat((part2_points, part2_points_), dim=0)
                part2_normals = torch.cat((part2_normals, part2_normals_), dim=0)
        if points.shape[0] < 10000 or part2_points.shape[0] < 5000:
            print("Part out of view")
            terminated_count += 1
            continue
        points = points.unsqueeze(0)
        fps_indexes = farthest_point_sample_GPU(points, 2048).squeeze(0)  # (2048)
        points = points.squeeze(0)
        points = points[fps_indexes, :]
        normals = normals[fps_indexes, :]
        part1_points = points
        part1_normals = normals

        part2_points = part2_points.unsqueeze(0)
        part2_fps_indexes = farthest_point_sample_GPU(part2_points, 2048).squeeze(0)
        part2_points = part2_points.squeeze(0)
        part2_points = part2_points[part2_fps_indexes, :]
        part2_normals = part2_normals[part2_fps_indexes, :]

        random_index = torch.randint(
            0, part1_points.shape[0], (1,), device=part1_points.device
        )

        sampled_point = part1_points[random_index].squeeze(0)  # (3)
        sampled_normal = part1_normals[random_index].squeeze(0)  # (3)

        result = sample_dir(sampled_point, sampled_normal)

        if result is False:
            print("Sample Hand Pose Collision")
            continue
        else:
            hand_pos, hand_ori, action_direct = result

        env.isaac_gym.end_access_image_tensors(env.sim)

        # Set hand pos
        try:
            env.set_hand_transform(hand_pos, hand_ori)
        except:
            print("Hand pos out of bound:", hand_pos, hand_ori)

            continue

        contact_flag = env.check_contact()
        # wait(env)
        if not contact_flag:

            # Start to perform the task
            module_name = (
                task_name if task_name.split("_")[-1] != "r" else task_name[:-2]
            )
            task_module = importlib.import_module(f"tasks.{module_name}")

            prepare = getattr(task_module, "prepare")
            if "drawer" in task_name and REV:
                prepare = getattr(task_module, "prepare_r")
            perform = getattr(task_module, "perform" if not REV else "perform_r")

            target_ee_states = None
            target_ee_states, result = prepare(env, target_ee_states)
            # check if the part1 is moved

            start_part1_pose = C.to_homogeneous(
                rb_states[part_idxs[part1_name]][0][:3],
                C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
            )
            start_part2_pose = C.to_homogeneous(
                rb_states[part_idxs[part2_name]][0][:3],
                C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
            )

            if not result:
                print("Gripper Collision at Pre Grasp XY")
                continue
            if not satisfy(ori_part1_pose, start_part1_pose, pos_error_threshold=0.05):
                print("Part1 Moved")
                continue
            if abs(ori_part2_pose[2, 3] - start_part2_pose[2, 3]) > 0.03:
                print("Part2 falled")
                continue

            env.sampled_points.append(sampled_point)
            env.sampled_normals.append(sampled_normal)
            env.action_directs.append(action_direct)

            env.set_last_check_step(max(len(env.observation[0]) - 1, 0))
            
            env.set_check(True)
            all_finished = True

            # Start screw

            all_finished = perform(env, target_ee_states)

            
            while env.need_reset:
                env.need_reset = False
                env.set_check(False)
                wait_close(env)
                move_franka_away(env)
                # conditioned = np.random.choice([True, False])
                conditioned =False
                if reset_hand_(env, show=False, conditioned=conditioned) is False:
                    print("Reset Hand Failed")
                    all_finished = False
                    break
                target_ee_states = None
                target_ee_states, result = prepare(env, target_ee_states)
                if not result:
                    all_finished = False
                    print("Gripper Collision at Prepare")
                    break

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
                final_part2_pose = C.to_homogeneous(
                    rb_states[part_idxs[part2_name]][0][:3],
                    C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
                )
                if (
                    abs(final_part1_pose[2, 3] - start_part1_pose[2, 3]) > 0.1
                    or abs(final_part2_pose[2, 3] - start_part2_pose[2, 3]) > 0.05
                ):
                    print("part out of table or fall")
                    continue
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
                        if index % 2 == 0:
                            continue
                        index = index // 2
                        sampled_point = env.sampled_points[index]
                        sampled_normal = env.sampled_normals[index]
                        cp2 = torch.cat((sampled_point, sampled_normal), dim=0)
                        dir2 = env.action_directs[index]
                        interaction["cp2"].append(cp2.cpu())
                        interaction["dir2"].append(dir2.cpu())
                        cur_step = env.moved[index * 2 + 1]
                        pre_step = env.moved[index * 2]
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
                        cur_T, final_part1_pose
                    )
                    points, normals = obs_to_pc_fps(
                        env.observation[0][env.moved[-1]],
                        env.camera_matix[0],
                        env.img_size,
                        [1, 2],
                    )
                    part1_points = points[1]
                    part1_normals = normals[1]
                    part2_points = points[2]
                    part2_normals = normals[2]
                print("Distance: ", distance)
                cur_data = {
                    "all_finished": all_finished,
                    "part1_points": part1_points.cpu(),
                    "part1_normals": part1_normals.cpu(),
                    "part2_points": part2_points.cpu(),
                    "part2_normals": part2_normals.cpu(),
                    "sampled_point": env.sampled_points[-1].cpu(),
                    "sampled_normal": env.sampled_normals[-1].cpu(),
                    "action_direct": env.action_directs[-1].cpu(),
                    "start_part1_pose": start_part1_pose.cpu(),
                    "final_part1_pose": final_part1_pose.cpu(),
                    "start_part2_pose": start_part2_pose.cpu(),
                    "final_part2_pose": final_part2_pose.cpu(),
                    "distance": distance.cpu(),
                    "interaction": interaction,
                }
                # save data
                if save:
                    if zip_save_finished:
                        finished_data.append(cur_data)
                    else:
                        save_path = os.path.join(
                            finished_path, "data_%d.pt" % finished_count
                        )
                        torch.save(cur_data, save_path)
                else:
                    write_data = np.array(
                        [
                            [
                                distance.item(),
                                final_part1_pose[0, 3].item(),
                                final_part1_pose[1, 3].item(),
                                final_part1_pose[2, 3].item(),
                                final_part2_pose[0, 3].item(),
                                final_part2_pose[1, 3].item(),
                                final_part2_pose[2, 3].item(),
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
                                "Distance,part1_pose_x,part1_pose_y,part1_pose_z,part2_pose_x,part2_pose_y,part2_pose_z"
                                if not file_exist
                                else ""
                            ),
                            fmt="%f",
                        )
                finished_count += 1
            else:
                terminated_count += 1
        else:
            if save_failed and save:
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
                    "part2_points": part2_points.cpu(),
                    "part2_normals": part2_normals.cpu(),
                    "sampled_point": sampled_point.cpu(),
                    "sampled_normal": sampled_normal.cpu(),
                    "action_direct": action_direct.cpu(),
                    "start_part1_pose": ori_part1_pose.cpu(),
                    "final_part1_pose": ori_part1_pose.cpu(),
                    "start_part2_pose": ori_part2_pose.cpu(),
                    "final_part2_pose": ori_part2_pose.cpu(),
                    "distance": distance,
                    "interaction": interaction,
                }
                if zip_save_finished:
                    finished_data.append(cur_data)
                else:
                    save_path = os.path.join(failed_path, "data_%d.pt" % failed_count)
                    torch.save(cur_data, save_path)
                failed_count += 1

        if finished_count >= 10000:
            break


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--task", type=str, default="drawer_top_1", help="task name")
    args = argparse.parse_args()
    # lamp|square_table|desk|drawer|cabinet|round_table|stool|chair|one_leg
    task_name = args.task
    task_config = all_task_config[task_name]

    collect_data(task_config)
