"""
Camera utility functions for processing observation data.
"""

from typing import Any

import numpy as np


def obs_get_rgb(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Recursively search through observation dictionary to find RGB images.

    Args:
        obs: Observation dictionary that may contain nested camera data

    Returns:
        Dictionary mapping camera names to RGB image arrays
    """
    rgb_dict = {}

    for key, value in obs.items():
        if isinstance(value, dict):
            # Check if this dict contains images with rgb data
            if "images" in value and isinstance(value["images"], dict):
                if "rgb" in value["images"]:
                    rgb_dict[key] = value["images"]["rgb"]
            else:
                # Recursively search in nested dictionaries
                nested_rgb = obs_get_rgb(value)
                rgb_dict.update(nested_rgb)

    return rgb_dict
