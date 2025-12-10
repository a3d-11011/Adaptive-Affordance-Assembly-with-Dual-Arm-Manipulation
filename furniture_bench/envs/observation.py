PC_OBS = [
    "pc",
    "hand_pose",
    "parts_poses",  
]
Dual_OBS=[
    "robot_state/ee_pos_l",
    "robot_state/ee_quat_l",
    "robot_state/ee_pos_vel_l",
    "robot_state/ee_ori_vel_l",
    "robot_state/gripper_width_l",
    "robot_state/ee_pos_r",
    "robot_state/ee_quat_r",
    "robot_state/ee_pos_vel_r",
    "robot_state/ee_ori_vel_r",
    "robot_state/gripper_width_r"
]
FULL_OBS = [
    "robot_state/ee_pos",
    "robot_state/ee_quat",
    "robot_state/ee_pos_vel",
    "robot_state/ee_ori_vel",
    "robot_state/gripper_width",
    "robot_state/joint_positions",
    "robot_state/joint_velocities",
    "robot_state/joint_torques",
    "color_image1",
    "depth_image1",
    "color_image2",
    "depth_image2",
    "color_image3",
    "depth_image3",
    "parts_poses",
]

DEFAULT_VISUAL_OBS = [
    "robot_state/ee_pos",
    "robot_state/ee_quat",
    "robot_state/ee_pos_vel",
    "robot_state/ee_ori_vel",
    "robot_state/gripper_width",
    "color_image2",
    "color_image3",
]

DEFAULT_STATE_OBS = [
    "robot_state/ee_pos",
    "robot_state/ee_quat",
    "robot_state/ee_pos_vel",
    "robot_state/ee_ori_vel",
    "robot_state/gripper_width",
    "parts_poses",
]
