"""Define task initialization modes for FurnitureBench."""
from enum import Enum


class InitializationMode(Enum):
    DISASSEMBLED = 0
    PREASSEMBLED = 1
    ASSEMBLED = 2


def str_to_init_mode(v):
    if isinstance(v, InitializationMode):
        return v
    if v == "disassembled" or v == "dis_assembled" or v == 0:
        return InitializationMode.DISASSEMBLED
    elif v == "preassembled" or v == "pre_assembled" or v == 1:
        return InitializationMode.PREASSEMBLED
    elif v == "assembled" or v == 2:
        return InitializationMode.ASSEMBLED
    else:
        raise ValueError(f"Unknown initialization mode: {v}")


class Randomness(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    MEDIUM_COLLECT = 3  # For data collection with medium-level perturbation.
    HIGH_COLLECT = 4
    SKILL_FIXED = 5
    SKILL_RANDOM = 6
    NONE = 7


def str_to_enum(v):
    if isinstance(v, Randomness):
        return v
    if v == "low":
        return Randomness.LOW
    elif v == "med":
        return Randomness.MEDIUM
    elif v == "high":
        return Randomness.HIGH
    elif v == "med_collect":
        return Randomness.MEDIUM_COLLECT
    elif v == "high_collect":
        return Randomness.HIGH_COLLECT
    elif v == "skill_fixed":
        return Randomness.SKILL_FIXED
    elif v == "skill_random":
        return Randomness.SKILL_RANDOM
    elif v == "none":
        return Randomness.NONE
    else:
        raise ValueError(f"Unknown initialization mode: {v}")
