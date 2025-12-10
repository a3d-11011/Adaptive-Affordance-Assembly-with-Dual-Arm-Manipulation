"""
Register FurnitureBench and FurnitureSim environments to OpenAI Gym.
"""

from gym.envs.registration import register

# Ignore ImportError from isaacgym.
try:
    import isaacgym
except ImportError:
    pass



register(
    id="dual-franka-hand-v2",
    entry_point="furniture_bench.envs.dual_franka_hand_m_env:FurnitureSimEnv",
)

