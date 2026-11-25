"""TRON2 integration package for CaP-X."""

from .official_client import (
    DEFAULT_HOME_Q16,
    HORIZONTAL_GRASP_QUAT_WXYZ,
    PROTOCOL_JOINT_NAMES,
    Tron2OfficialClient,
)

__all__ = [
    "DEFAULT_HOME_Q16",
    "HORIZONTAL_GRASP_QUAT_WXYZ",
    "PROTOCOL_JOINT_NAMES",
    "Tron2OfficialClient",
]


