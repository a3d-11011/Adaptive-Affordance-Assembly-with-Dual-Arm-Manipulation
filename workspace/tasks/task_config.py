import os
import numpy as np
from typing import Any, Dict

BASE_DIR = os.path.dirname(__file__)

all_task_config: Dict[str, Any] = {
    "square_table": {
        "task_name": "square_table",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg4"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR,
            "relative_poses/square_table_square_table_top_square_table_leg4_fine.npy",
        ),
        "part_frictions": [0.08, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "square_table_1": {
        "task_name": "square_table_1",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg1"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR,
            "relative_poses/square_table_square_table_top_square_table_leg1_fine.npy",
        ),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "square_table_2": {
        "task_name": "square_table_2",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg2"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR,
            "relative_poses/square_table_square_table_top_square_table_leg2_fine.npy",
        ),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "square_table_3": {
        "task_name": "square_table_3",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg3"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR,
            "relative_poses/square_table_square_table_top_square_table_leg3_fine.npy",
        ),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "square_table_r": {
        "task_name": "square_table_r",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg4"],
        "part_frictions": [0.08, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "square_table_1_r": {
        "task_name": "square_table_1_r",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg1"],
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "square_table_2_r": {
        "task_name": "square_table_2_r",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg2"],
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "square_table_3_r": {
        "task_name": "square_table_3_r",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg3"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR,
            "relative_poses/square_table_square_table_top_square_table_leg3.npy",
        ),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "chair": {
        "task_name": "chair",
        "furniture_name": "chair",
        "part_names": ["chair_seat", "chair_leg1"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/chair_chair_seat_chair_leg1_fine.npy"
        ),
        "part_frictions": [0.1, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.15, -0.1), (0.1, 0.15), (-0.08, -0.08)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 2.5,
    },
    "chair_1": {
        "task_name": "chair_1",
        "furniture_name": "chair",
        "part_names": ["chair_seat", "chair_leg2"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/chair_chair_seat_chair_leg2_fine.npy"
        ),
        "part_frictions": [0.1, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.08, -0.04),(0.1, 0.15), (-0.08, -0.08)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 2.5,
    },
    "chair_r": {
        "task_name": "chair_r",
        "furniture_name": "chair",
        "part_names": ["chair_seat", "chair_leg1"],
        "part_frictions": [0.1, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
             "pos": [(-0.08, -0.04), (0.1, 0.15), (-0.08, -0.08)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 2.5,
    },
    "chair_1_r": {
        "task_name": "chair_1_r",
        "furniture_name": "chair",
        "part_names": ["chair_seat", "chair_leg2"],
        "part_frictions": [0.1, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.08, -0.04), (0.1, 0.15), (-0.08, -0.08)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 2.5,
    },
    "lamp": {
        "task_name": "lamp",
        "furniture_name": "lamp",
        "part_names": ["lamp_base", "lamp_bulb"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/lamp_lamp_base_lamp_bulb.npy"
        ),
        "part_frictions": [0.08, 0.5],
        "part_mass": [1, 0.5],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (-0.05, -0.05)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 2.0,
    },
    "desk": {
        "task_name": "desk",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg4"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/desk_desk_top_desk_leg4.npy"
        ),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (0.02, 0.025), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "desk_1": {
        "task_name": "desk_1",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg1"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/desk_desk_top_desk_leg1.npy"
        ),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (0.02, 0.025), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "desk_2": {
        "task_name": "desk_2",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg2"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/desk_desk_top_desk_leg2.npy"
        ),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            # "pos": [(0.15, 0.15), (0.06, 0.06), (-0.05, -0.05)],
            # "pos": [(-0.0, 0.0), (0.0, 0.0), (0, 0)],
            "pos": [(-0.02, 0.02), (0.02, 0.03), (-0.05, -0.05)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],#(np.pi+np.pi/4, np.pi+np.pi/4)
        },
        "scale_factor": 1.5,
    },
    "desk_3": {
        "task_name": "desk_3",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg3"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/desk_desk_top_desk_leg3.npy"
        ),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (0.015, 0.025), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "desk_triangle_testleg": {
        "task_name": "desk_triangle_testleg",
        "furniture_name": "desk_triangle",
        "part_names": ["desk_top_triangle", "desk_leg1"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR,
            "relative_poses/desk_desk_top_desk_leg1.npy",
            # "relative_poses/desk_triangle_desk_top_triangle_desk_leg3.npy"
        ),
        "part_frictions": [0.1, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(0.05, 0.05), (0.02, 0.02), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.75,
    },
    "desk_triangle_testleg_r": {
        "task_name": "desk_triangle_testleg_r",
        "furniture_name": "desk_triangle",
        "part_names": ["desk_top_triangle", "desk_leg1"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR,
            "relative_poses/desk_desk_top_desk_leg1.npy",
            # "relative_poses/desk_triangle_desk_top_triangle_desk_leg3.npy"
        ),
        "part_frictions": [0.1, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(0.05, 0.05), (0.02, 0.02), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.75,
    },
    "desk_rectangle_testleg": {
        "task_name": "desk_rectangle_testleg",
        "furniture_name": "desk_rectangle",
        "part_names": ["desk_top_rectangle", "desk_leg4"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/desk_rectangle_desk_top_rectangle_desk_leg4.npy"
        ),
        "part_frictions": [0.08, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, -0.02), (0.0, 0.00), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi / 3, np.pi / 3)],
        },
        "scale_factor": 1.5,
    },
    "desk_rectangle_testleg_r": {
        "task_name": "desk_rectangle_testleg_r",
        "furniture_name": "desk_rectangle",
        "part_names": ["desk_top_rectangle", "desk_leg4"],
        "part_frictions": [0.1, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, -0.02), (0.0, 0.00), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi / 3, np.pi / 3)],
        },
        "scale_factor": 1.5,
    },
    "desk_trapezoid_testleg": {
        "task_name": "desk_trapezoid_testleg",
        "furniture_name": "desk_trapezoid",
        "part_names": ["desk_top_trapezoid", "desk_leg1"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/desk_desk_top_desk_leg1.npy"
        ),
        "part_frictions": [0.1, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(0.05, 0.05), (0.05, 0.05), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "desk_r": {
        "task_name": "desk_r",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg4"],
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (0.0, 0.025), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "desk_1_r": {
        "task_name": "desk_1_r",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg1"],
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (0.0, 0.025), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "desk_2_r": {
        "task_name": "desk_2_r",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg2"],
        # "disassembled_pose_path": os.path.join(BASE_DIR,"relative_poses/desk_desk_top_desk_leg2.npy"),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (0.0, 0.025), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "desk_3_r": {
        "task_name": "desk_3_r",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg3"],
        # "disassembled_pose_path": os.path.join(BASE_DIR,"relative_poses/desk_desk_top_desk_leg3.npy"),
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (0.0, 0.025), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "drawer_top": {
        "task_name": "drawer_top",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_container_top"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/drawer_drawer_box_drawer_container_top.npy"
        ),
        "part_frictions": [0.05, 0.3],
        "randomness": {
            "pos": [(-0.08, -0.05), (-0.05, 0.05), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6, np.pi / 6)],
        },
        "scale_factor": 2.0,
    },
    "drawer_top_1": {
        "task_name": "drawer_top_1",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_top_1"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/drawer_drawer_box_drawer_container_top.npy"
        ),
        "part_frictions": [0.05, 0.3],
        "randomness": {
            "pos": [(-0.08, -0.05), (-0.05, 0.05), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6, np.pi / 6)],
        },
        "scale_factor": 2.0,
    },
    "drawer_top_2": {
        "task_name": "drawer_top_2",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_top_2"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/drawer_drawer_box_drawer_container_top.npy"
        ),
        "part_frictions": [0.05, 0.5],
        "randomness": {
            "pos": [(-0.08, -0.05), (-0.05, 0.05), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (0, 0)],
        },
        "scale_factor": 2.0,
    },
    "drawer_bottom": {
        "task_name": "drawer_bottom",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_container_bottom"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/drawer_drawer_box_drawer_container_bottom.npy"
        ),
        "part_frictions": [0.05, 0.3],
        "randomness": {
            "pos": [(-0.08, -0.05), (-0.05, 0.05), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6, np.pi / 6)],
        },
        "scale_factor": 2.0,
    },
    "drawer_bottom_1": {
        "task_name": "drawer_bottom_1",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_bottom_1"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/drawer_drawer_box_drawer_container_bottom.npy"
        ),
        "part_frictions": [0.05, 0.3],
        "randomness": {
            "pos": [(-0.08, -0.05), (-0.05, 0.05), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6, np.pi / 6)],
        },
        "scale_factor": 2.0,
    },
    "drawer_bottom_2": {
        "task_name": "drawer_bottom_2",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_bottom_2"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/drawer_drawer_box_drawer_container_bottom.npy"
        ),
        "part_frictions": [0.05, 0.3],
        "randomness": {
            "pos": [(-0.08, -0.05), (-0.05, 0.05), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6, np.pi / 6)],
        },
        "scale_factor": 2.0,
    },
    "cabinet_door_left": {
        "task_name": "cabinet_door_left",
        "furniture_name": "cabinet",
        "part_names": ["cabinet_body", "cabinet_door_left"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/cabinet_cabinet_body_cabinet_door_left.npy"
        ),
        "part_frictions": [0.1, 0.6],
        # "part_mass": [1, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (-0.05, -0.05)],
            "ori": [(0, 0), (0, 0), ( np.pi / 6, np.pi / 6)],
            # "ori": [(0,0), (0,0), (0,0)]
        },
        "scale_factor": 2,
    },
    "cabinet_door_right": {
        "task_name": "cabinet_door_right",
        "furniture_name": "cabinet",
        "part_names": ["cabinet_body", "cabinet_door_right"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR,
            "relative_poses/cabinet_cabinet_body_cabinet_door_right_refine.npy",
        ),
        "part_frictions": [0.1, 0.6],
        # "part_mass": [1, 0.25],
        "randomness": {
            "pos": [(0.02, 0.04), (-0.03, 0.01), (-0.05, -0.05)],
            "ori": [(0, 0), (0, 0), ( - np.pi / 6,  np.pi / 6)],
            # "ori": [(0,0), (0,0), (0,0)]
        },
        "scale_factor": 2,
    },
    "desk_multi": {
        "task_name": "desk_multi",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg4", "desk_leg1", "desk_leg2", "desk_leg3"],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/desk_desk_top_desk_leg4.npy"
        ),
        "part_frictions": [0.03, 2, 2, 2, 2, 2],
        "part_mass": [0.5, 0.25, 0.25, 0.25, 0.25],
        "randomness": {
            "pos": [(-0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi, np.pi)],
        },
        "scale_factor": 1.5,
    },
    "drawer_top_r": {
        "task_name": "drawer_top_r",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_container_top", "drawer_container_bottom"],
        "part_frictions": [0.2, 2.0, 2.0],
        "part_mass": [1, 0.5, 0.5],
        "randomness": {
            "pos": [(-0.12, -0.09), (-0.05, 0.05), (-0.05, -0.05)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6 + np.pi / 36, np.pi / 36)],
        },
        "scale_factor": 2.5,
    },
    "drawer_top_1_r": {
        "task_name": "drawer_top_1_r",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_top_1", "drawer_container_bottom"],
        "part_frictions": [0.2, 2.0, 2.0],
        "part_mass": [1, 0.5, 0.5],
        "randomness": {
            "pos": [(-0.12, -0.09), (-0.05, 0.05), (-0.05, -0.05)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6 + np.pi / 36, np.pi / 36)],
        },
        "scale_factor": 2.5,
    },
    "drawer_top_2_r": {
        "task_name": "drawer_top_2_r",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_top_2", "drawer_container_bottom"],
        "part_frictions": [0.2, 2.0, 2.0],
        "part_mass": [1, 0.5, 0.5],
        "randomness": {
            "pos": [(-0.12, -0.09), (0.1, 0.2), (-0.05, -0.05)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6 + np.pi / 36, np.pi / 36)],
        },
        "scale_factor": 2.5,
    },
    "drawer_bottom_r": {
        "task_name": "drawer_bottom_r",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_container_bottom", "drawer_container_top"],
        "part_frictions": [0.2, 2.0, 2.0],
        "part_mass": [1, 0.5, 0.5],
        "randomness": {
            "pos": [(-0.12, -0.09), (0.1, 0.2), (-0.07, -0.07)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6 + np.pi / 36, np.pi / 36)],
        },
        "scale_factor": 2.5,
        "box": {
            "size": [1, 1, 0.02],
        },
    },
    "drawer_bottom_1_r": {
        "task_name": "drawer_bottom_1_r",
        "furniture_name": "drawer",
        "part_names": [
            "drawer_box",
            "drawer_bottom_1",
            "drawer_container_top",
        ],
        "part_frictions": [0.2, 2.0, 2.0],
        "part_mass": [1, 0.5, 0.5],
        "randomness": {
            "pos": [(-0.12, -0.09), (0.1, 0.2), (-0.07, -0.07)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6 + np.pi / 36, np.pi / 36)],
        },
        "scale_factor": 2.5,
        "box": {
            "size": [1, 1, 0.02],
        },
    },
    "drawer_bottom_2_r": {
        "task_name": "drawer_bottom_2_r",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_bottom_2", "drawer_container_top"],
        "part_frictions": [0.2, 2.0, 2.0],
        "part_mass": [1, 0.5, 0.5],
        "randomness": {
            "pos": [(-0.12, -0.09), (0.1, 0.2), (-0.07, -0.07)],
            "ori": [(0, 0), (0, 0), (-np.pi / 6 + np.pi / 36, np.pi / 36)],
        },
        "scale_factor": 2.5,
        "box": {
            "size": [1, 1, 0.02],
        },
    },
    "desk_1_pull": {
        "task_name": "desk_1_pull",
        "furniture_name": "desk_pull",
        "part_names": ["desk_top", "desk_leg1", "desk_leg2", "desk_leg3"],
        # "disassembled_pose_path":  os.path.join(BASE_DIR,"relative_poses/desk_pull_desk_top_desk_leg1.npy"),
        "part_frictions": [2, 0.5, 1, 1],
        "part_mass": [1, 0.25, 0.25, 0.25],
        "randomness": {
            "pos": [(-0.0, -0.0), (0.0, 0.0), (-0.2, -0.2)],
            "ori": [(0, 0), (0, 0), (0, 0)],
        },
        "scale_factor": 2.0,
        "box": {
            "size": [1, 0.5, 0.1],
        },
    },
    "desk_2_pull": {
        "task_name": "desk_2_pull",
        "furniture_name": "desk_pull",
        "part_names": ["desk_top", "desk_leg2", "desk_leg1", "desk_leg3"],
        # "disassembled_pose_path":  os.path.join(BASE_DIR,"relative_poses/desk_pull_desk_top_desk_leg1.npy"),
        "part_frictions": [2, 0.5, 1, 1],
        "part_mass": [1, 0.25, 0.25, 0.25],
        "randomness": {
            "pos": [(-0.0, -0.0), (0.05, 0.05), (-0.2, -0.2)],
            "ori": [(0, 0), (0, 0), (0, 0)],
        },
        "scale_factor": 2.0,
        "box": {
            "size": [1, 0.5, 0.1],
        },
    },
    "desk_3_pull": {
        "task_name": "desk_3_pull",
        "furniture_name": "desk_pull",
        "part_names": ["desk_top", "desk_leg3", "desk_leg1", "desk_leg2"],
        # "disassembled_pose_path":  os.path.join(BASE_DIR,"relative_poses/desk_pull_desk_top_desk_leg1.npy"),
        "part_frictions": [2, 0.5, 1, 1],
        "part_mass": [1, 0.25, 0.25, 0.25],
        "randomness": {
            "pos": [(-0.08, -0.08), (-0.0, -0.0), (-0.2, -0.2)],
            "ori": [(0, 0), (0, 0), (0, 0)],
        },
        "scale_factor": 2.0,
        "box": {
            "pos": [0.0, -0.1, 0.0],
            "size": [1, 0.5, 0.1],
        },
    },
    "bucket": {
        "task_name": "bucket",
        "furniture_name": "bucket",
        "part_names": ["bucket_body"],
        "part_frictions": [3.0],
        "part_mass": [1.0],
        "randomness": {
            "pos": [(-0.06, -0.04), (0.0, 0.04), (-0.1, -0.1)],
            "ori": [(0, 0), (0, 0), (-np.pi / 4 + np.pi / 2, -np.pi / 4 + np.pi / 2)],
        },
        "scale_factor": 2.0,
        # "box": {"size": [0.8, 0.8, 0.05], "pos": [0.25, 0.0, 0.0]},
    },
    "round_table": {
        "task_name": "round_table",
        "furniture_name": "round_table",
        "part_names": ["round_table_top", "round_table_leg"],
        "part_frictions": [0.08, 2, 2],
        "part_mass": [0.5, 0.25, 0.25],
        "randomness": {
            "pos": [(0.02, 0.02), (-0.02, 0.02), (0, 0)],
            "ori": [(0, 0), (0, 0), (-np.pi / 36, -np.pi / 36)],
        },
        "scale_factor": 1.5,
    },
    "oval_table": {
        "task_name": "oval_table",
        "furniture_name": "oval_table",
        "part_names": ["oval_table_top", "oval_table_leg"],
        "part_frictions": [0.1, 1, 0.5],
        "disassembled_pose_path": os.path.join(
            BASE_DIR, "relative_poses/oval_table_oval_table_top_oval_table_leg.npy"
        ),
        "part_mass": [0.5, 0.25, 0.25],
        "randomness": {
            "pos": [(0.05, 0.05), (0.05, 0.05), (-0.02, -0.02)],
            "ori": [(0, 0), (0, 0), (0, 0)],
        },
        "scale_factor": 1.5,
    },
    "drawer_c": {
        "task_name": "drawer_c",
        "furniture_name": "drawer",
        "part_names": ["drawer_container_top"],
        "part_frictions": [3.0],
        "part_mass": [1],
        "randomness": {
            "pos": [(-0.06-0.15, -0.04-0.15), (-0.15, -0.14), (-0.06, -0.06)],
            "ori": [(0, 0), (0, 0), (-np.pi/2-np.pi/18,-np.pi/2+np.pi/18)],
        },
        "scale_factor": 3.0,
    },
    "cask": {
        "task_name": "cask",
        "furniture_name": "cask",
        "part_names": ["cask_body"],
        "part_frictions": [3.0],
        "part_mass": [1.0],
        "randomness": {
            "pos": [(-0.06, -0.04), (0.0, 0.04), (-0.1, -0.1)],
            "ori": [(0, 0), (0, 0), (-np.pi / 4 + np.pi / 2, np.pi / 4 + np.pi / 2)],
        },
        "scale_factor": 1.0,
        # "box": {"size": [0.8, 0.8, 0.05], "pos": [0.25, 0.0, 0.0]},
    },
}
