import os
import sys
import shutil
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.distance import (
    double_geodesic_distance_between_poses,
    geodesic_distance_between_R,
)


def filter_success_data(data_dir, dst_dir, threshold=0.1, task="desk"):
    data_list = os.listdir(data_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    success_ = 0
    for data_ in data_list:
        try:
            finished_data = torch.load(os.path.join(data_dir, data_), weights_only=True)
        except:
            print(f"Failed to load {data_}")
            raise ("Failed to load data")

        if "distance" in finished_data.keys():
            dist = finished_data["distance"]
        else:
            dist = double_geodesic_distance_between_poses(
                finished_data["final_part1_pose"], finished_data["start_part1_pose"]
            )

        if "drawer" in task:
            part_dis = torch.norm(
                finished_data["final_part2_pose"][:2, 3]
                - finished_data["final_part1_pose"][:2, 3]
            )
            if "pull" in task:
                dist += max(0, 0.2 - part_dis) * 2
            else:
                # dist+= part_dis*2
                dist = 1 - ((0.2 - part_dis) * 5 / 2 + (0.5 - dist))
        elif "bucket" in task:
            height = (
                finished_data["final_part1_pose"][2, 3]
                - finished_data["start_part1_pose"][2, 3]
            )
            dist = (0.14 - height) * 10 + geodesic_distance_between_R(
                finished_data["start_part1_pose"][:3, :3],
                finished_data["final_part1_pose"][:3, :3],
            ) * 10

        if dist < threshold:
            shutil.copy(
                os.path.join(data_dir, data_),
                os.path.join(dst_dir, f"data_{success_}.pt"),
            )
            success_ += 1
        else:
            print(f"Filtered out {data_},distance={dist}")


def filter_actor_data(data_dir, dst_dir, threshold=0.1, portion=1):
    data_list = os.listdir(data_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    success_ = 0
    failed_ = 0
    for data_ in data_list:
        finished_data = torch.load(os.path.join(data_dir, data_), weights_only=True)
        dist = double_geodesic_distance_between_poses(
            finished_data["final_part1_pose"], finished_data["start_part1_pose"]
        )
        if dist < threshold:
            shutil.copy(
                os.path.join(data_dir, data_),
                os.path.join(dst_dir, f"data_{success_+failed_}.pt"),
            )
            success_ += 1
        else:
            if failed_ < success_ * portion:
                shutil.copy(
                    os.path.join(data_dir, data_),
                    os.path.join(dst_dir, f"data_{success_+failed_}.pt"),
                )
                failed_ += 1
    print(f"success={success_},failed={failed_}")
    print(f"portion={failed_/success_}")


def filter_interact_data(data_dir, dst_dir):
    data_list = os.listdir(data_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    success_ = 0
    for data_ in data_list:
        finished_data = torch.load(os.path.join(data_dir, data_), weights_only=True)
        if len(finished_data["interaction"]["move"]) != 0:
            shutil.copy(
                os.path.join(data_dir, data_),
                os.path.join(dst_dir, f"data_{success_}.pt"),
            )
            success_ += 1
        else:
            print(f"Filtered out {data_}")
    print(f"success={success_}")


def filter_wrong_distance_data(data_dir):
    data_list = os.listdir(data_dir)
    for data_ in data_list:
        finished_data = torch.load(os.path.join(data_dir, data_), weights_only=True)
        if finished_data["distance"] == torch.tensor([100.0]):

            print(f"File {data_} has a distance of [100.0], changing it to 100.0")


def filter_distance_data(data_dir, dst_dir):
    data_list = os.listdir(data_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    success_ = 0
    for data_ in data_list:
        finished_data = torch.load(os.path.join(data_dir, data_), weights_only=True)
        if finished_data["distance"] != torch.tensor(100.0) and finished_data[
            "distance"
        ] != torch.tensor([100.0]):
            shutil.copy(
                os.path.join(data_dir, data_),
                os.path.join(dst_dir, f"data_{success_}.pt"),
            )
            success_ += 1
        else:
            print(f"Filtered out {data_}")
    print(f"success={success_}")


def filter_list_data(data_dir):
    data_list = os.listdir(data_dir)
    for data_ in data_list:
        finished_data = torch.load(os.path.join(data_dir, data_), weights_only=True)
        if finished_data["distance"] == torch.tensor([100.0]):
            finished_data["distance"] = torch.tensor(100.0)
            torch.save(finished_data, os.path.join(data_dir, data_))
            print("changed", data_)


filter_success_data(
    "/mnt/data/Dual-assemble/data_cl/bucket_full/finished",
    "/mnt/data/Dual-assemble/data_cl_s/bucket_full/finished",
    threshold=0.2,
    task="bucket",
)
